import sys, os

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join(now_dir, "train"))
import utils
import datetime

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import shuffle, randint
import traceback, json, argparse, itertools, math, torch, pdb
from tqdm import tqdm # Import tqdm

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from lib.infer_pack import commons
from time import sleep
from time import time as ttime
from data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)

import csv

if hps.version == "v1":
    from lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid as RVC_Model_f0,
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminator,
    )
else:
    from lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(
                i,
                n_gpus,
                hps,
            ),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # Set find_unused_parameters=True to handle freezing/unfreezing
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    else:
         # Set find_unused_parameters=True for CPU as well if using DDP
        net_g = DDP(net_g, find_unused_parameters=True)
        net_d = DDP(net_d, find_unused_parameters=True)


    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            # Using strict=False to allow loading checkpoints with different parameter sets
            try:
                net_g.module.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu")["model"], strict=False
                )
            except RuntimeError as e:
                 if rank == 0:
                     logger.warning(f"Partial loading of pretrained G model: {e}")
                 # Attempt partial load if strict=False fails
                 try:
                     net_g.module.load_state_dict(
                         torch.load(hps.pretrainG, map_location="cpu")["model"], strict=False
                     )
                 except Exception as e_partial:
                     if rank == 0:
                         logger.error(f"Failed to load pretrained G model even partially: {e_partial}")

        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            # Using strict=False to allow loading checkpoints with different parameter sets
            try:
                net_d.module.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu")["model"], strict=False
                )
            except RuntimeError as e:
                 if rank == 0:
                     logger.warning(f"Partial loading of pretrained D model: {e}")
                 # Attempt partial load if strict=False fails
                 try:
                     net_d.module.load_state_dict(
                         torch.load(hps.pretrainD, map_location="cpu")["model"], strict=False
                     )
                 except Exception as e_partial:
                     if rank == 0:
                         logger.error(f"Failed to load pretrained D model even partially: {e_partial}")


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # --- Freezing/Unfreezing Logic based on Global Step ---
    current_step = global_step

    # Determine which layers to unfreeze based on the current step
    unfreeze_layers = []
    step_range_description = ""

    if current_step < 300:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post']
        step_range_description = "0-300: Tune speaker identity and projection"
    elif 300 <= current_step < 600:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post',
                           'dec.ups.3', 'dec.resblocks.9', 'dec.resblocks.10', 'dec.resblocks.11']
        step_range_description = "300-600: Start shaping voice with low-capacity decoder"
    elif 600 <= current_step < 1000:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post',
                           'dec.ups.3', 'dec.resblocks.9', 'dec.resblocks.10', 'dec.resblocks.11',
                           'dec.ups.2', 'dec.resblocks.6', 'dec.resblocks.7', 'dec.resblocks.8']
        step_range_description = "600-1000: Gradually increase decoder expressiveness"
    elif 1000 <= current_step < 1400:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post',
                           'dec.ups.3', 'dec.resblocks.9', 'dec.resblocks.10', 'dec.resblocks.11',
                           'dec.ups.2', 'dec.resblocks.6', 'dec.resblocks.7', 'dec.resblocks.8']
        for i in range(3):
             unfreeze_layers.append(f'encoder.attn_layers.{i}')
             unfreeze_layers.append(f'encoder.norm_layers_1.{i}')
        unfreeze_layers.append('encoder.ffn_layers.0')
        step_range_description = "1000-1400: Begin adapting encoder attention"
    elif 1400 <= current_step < 1800:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post',
                           'dec.ups.3', 'dec.resblocks.9', 'dec.resblocks.10', 'dec.resblocks.11',
                           'dec.ups.2', 'dec.resblocks.6', 'dec.resblocks.7', 'dec.resblocks.8']
        for i in range(6):
             unfreeze_layers.append(f'encoder.attn_layers.{i}')
             unfreeze_layers.append(f'encoder.norm_layers_1.{i}')
        for i in range(6):
             unfreeze_layers.append(f'encoder.ffn_layers.{i}')
        unfreeze_layers.append('encoder.proj')
        step_range_description = "1400-1800: Fully adapt encoder"
    elif 1800 <= current_step < 2300:
        unfreeze_layers = ['emb_g', 'enc_q.proj', 'dec.conv_post',
                           'dec.ups.3', 'dec.resblocks.9', 'dec.resblocks.10', 'dec.resblocks.11',
                           'dec.ups.2', 'dec.resblocks.6', 'dec.resblocks.7', 'dec.resblocks.8']
        for i in range(6):
             unfreeze_layers.append(f'encoder.attn_layers.{i}')
             unfreeze_layers.append(f'encoder.norm_layers_1.{i}')
        for i in range(6):
             unfreeze_layers.append(f'encoder.ffn_layers.{i}')
        unfreeze_layers.append('encoder.proj')
        for i in range(2):
            unfreeze_layers.append(f'dec.ups.{i}')
        for i in range(6):
            unfreeze_layers.append(f'dec.resblocks.{i}')
        unfreeze_layers.append('dec.noise_convs')
        step_range_description = "1800-2300: Release full decoder power"
    else: # current_step >= 2300
        unfreeze_layers = None # Unfreeze all
        step_range_description = "2300+: All layers unfrozen for joint tuning"


    total_params = sum(p.numel() for p in net_g.parameters())
    trainable_params = 0

    # Apply freezing/unfreezing
    if unfreeze_layers is None:
        # Unfreeze all parameters
        for name, param in net_g.named_parameters():
             param.requires_grad = True
             trainable_params += param.numel()
        if rank == 0:
             logger.info(f"Step {current_step}: {step_range_description}. Unfreezing ALL layers.")
    else:
        # Freeze all parameters initially
        for param in net_g.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        for name, param in net_g.named_parameters():
            should_unfreeze = False
            for layer_name in unfreeze_layers:
                if layer_name in name:
                    should_unfreeze = True
                    break
            if should_unfreeze:
                param.requires_grad = True
                trainable_params += param.numel()

        if rank == 0:
            logger.info(f"Step {current_step}: {step_range_description}. Unfreezing layers: {unfreeze_layers}")

    # Log trainable parameter percentage
    trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
    if rank == 0:
        logger.info(f"Step {current_step}: Trainable parameters: {trainable_params}/{total_params} ({trainable_percentage:.2f}%)")

    # Ensure discriminator is always trainable
    for param in net_d.parameters():
        param.requires_grad = True
    # --- End Freezing/Unfreezing Logic ---


    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Wrap data_iterator with tqdm for progress bar
    if rank == 0:
        data_iterator = tqdm(data_iterator, desc=f"Epoch {epoch}", leave=False)


    # Run steps
    epoch_recorder = EpochRecorder()

    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # Update tqdm description with loss information
                data_iterator.set_postfix(
                    loss_g=loss_gen_all.item(),
                    loss_d=loss_disc.item(),
                    lr=lr,
                    step=global_step
                )
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # Save standalone G checkpoint every 150 steps
            if global_step % 150 == 0 and global_step > 0:
                 if hasattr(net_g, "module"):
                     ckpt = net_g.module.state_dict()
                 else:
                     ckpt = net_g.state_dict()
                 logger.info(
                     "saving standalone G ckpt %s_s%s:%s"
                     % (
                         hps.name,
                         global_step,
                         savee(
                             ckpt,
                             hps.sample_rate,
                             hps.if_f0,
                             hps.name + "_s%s" % (global_step),
                             epoch, # Still log the current epoch
                             hps.version,
                             hps,
                         ),
                     )
                 )

        global_step += 1
    # /Run steps

    # Save full checkpoints (G, D, optimizers) every hps.save_every_epoch
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            # Save as latest checkpoint
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_latest.pth"),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_latest.pth"),
            )

        # Save standalone G checkpoint every hps.save_every_epoch as well (optional, but good practice)
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s_s%s:%s"
                % (
                    hps.name,
                    epoch,
                    global_step,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )


    try:
        with open("csvdb/stop.csv") as CSVStop:
            csv_reader = list(csv.reader(CSVStop))
            stopbtn = (
                csv_reader[0][0]
                if csv_reader is not None
                else (lambda: exec('raise ValueError("No data")'))()
            )
            stopbtn = (
                lambda stopbtn: True
                if stopbtn.lower() == "true"
                else (False if stopbtn.lower() == "false" else stopbtn)
            )(stopbtn)
    except (ValueError, TypeError, IndexError):
        stopbtn = False

    if stopbtn:
        logger.info("Stop Button was pressed. The program is closed.")
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt,
                    hps.sample_rate,
                    hps.if_f0,
                    hps.name,
                    epoch,
                    hps.version,
                    hps,
                )
            )
        )
        sleep(1)
        with open("csvdb/stop.csv", "w+", newline="") as STOPCSVwrite:
            csv_writer = csv.writer(STOPCSVwrite, delimiter=",")
            csv_writer.writerow(["False"])
        os._exit(2333333)

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        with open("csvdb/stop.csv", "w+", newline="") as STOPCSVwrite:
            csv_writer = csv.writer(STOPCSVwrite, delimiter=",")
            csv_writer.writerow(["False"])
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()