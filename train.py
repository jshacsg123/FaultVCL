import argparse
import os
import logging
from time import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.head import Head
from optimizers.lr_scheduler import WarmupCosineSchedule
from utils.seisicdata_utils import seis_dataset
from utils.ops import concat_image
from utils.utils import AverageMeter

cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy("file_system")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "28890"


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def train(args, global_step, train_loader, val_best, scaler):
    model.train()
    loss_train = []
    run_loss = AverageMeter()
    scaleA_avg, scaleB_avg = AverageMeter(), AverageMeter()

    for step, batch in enumerate(train_loader):
        t1 = time()
        img, labels, crops = batch

        # imgs / crops are list[list(dict)], flatten them using concat_image from the project
        img, crops = concat_image(img), concat_image(crops)

        # ---- Important: labels shape fix ----
        # Typical cases: labels.shape = [B, sw, K] or [B, sw, 1, K]
        if hasattr(labels, "dim"):
            if labels.dim() == 3:
                # [B, sw, K] -> take the first view label [B, K]
                labels = labels[:, 0, :]
            elif labels.dim() == 4:
                # [B, sw, 1, K] -> [B, K]
                labels = labels[:, 0, 0, :]
            elif labels.dim() == 2:
                # already [B, K], use as is
                pass
            else:
                raise ValueError(f"Unexpected labels shape: {labels.shape}")

        img, crops, labels = img.cuda(), crops.cuda(), labels.cuda()

        with autocast(enabled=args.amp):
            # VoCoHead returns loss components from two scales
            L_A, L_B = model(img, crops, labels)

            # DataParallel may return tensors with extra dims, reduce to scalar
            if L_A.dim() > 0:
                L_A = L_A.mean()
            if L_B.dim() > 0:
                L_B = L_B.mean()

            # Total loss: L_total = L^A + beta_dual * L^B
            loss = L_A + args.beta_dual * L_B
            loss_train.append(loss.item())

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.lrdecay and scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        run_loss.update(loss.item(), n=args.batch_size)
        scaleA_avg.update(L_A.item(), n=args.batch_size)
        scaleB_avg.update(L_B.item(), n=args.batch_size)

        lr = optimizer.param_groups[0]["lr"]

        if args.distributed:
            if dist.get_rank() == 0:
                print(
                    "Step:{}/{}, Loss:{:.4f}, L_A:{:.4f}, L_B:{:.4f}, lr:{:.8f}, Time:{:.4f}".format(
                        global_step,
                        args.num_steps,
                        run_loss.avg,
                        scaleA_avg.avg,
                        scaleB_avg.avg,
                        lr,
                        time() - t1,
                    )
                )
        else:
            log_message = (
                "Step:{}/{}, Loss:{:.4f}, L_A:{:.4f}, L_B:{:.4f}, "
                "lr:{:.8f}, Time:{:.4f}".format(
                    global_step,
                    args.num_steps,
                    run_loss.avg,
                    scaleA_avg.avg,
                    scaleB_avg.avg,
                    lr,
                    time() - t1,
                )
            )
            print(log_message)
            with open("training_logs.txt", "a") as f:
                f.write(log_message + "\n")

        global_step += 1

        if args.distributed:
            val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
        else:
            val_cond = global_step % args.eval_num == 0

        freq = 2000
        val_freq = global_step % freq == 0

        if val_cond:
            checkpoint = {
                "global_step": global_step,
                "state_dict": model.state_dict(),
                "optimizer": optimizer,
            }
            save_ckp(checkpoint, os.path.join(logdir, "model_current_epoch.pt"))

        if val_freq:
            checkpoint = {
                "global_step": global_step,
                "state_dict": model.state_dict(),
                "optimizer": optimizer,
            }
            save_ckp(checkpoint, os.path.join(logdir, f"model_step{global_step}.pt"))

    return global_step, loss, val_best


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return logging.getLogger(name)
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    global model, optimizer, scheduler, logdir

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--field_dir",
        type=str,
        default=r"G:/field_data_withoutlabels",
        help="Directory containing all .npy 3D seismic volumes (default: data/)",
    )

    # VoCo cropping parameters for small and large views
    parser.add_argument(
        "--roi_small",
        type=int,
        default=32,
        help="Side length of the cropped small view (e.g., 32³), corresponds to args.roi_small",
    )
    parser.add_argument(
        "--roi_big",
        type=int,
        default=32,
        help="Side length of the randomly cropped large view (e.g., 32³), corresponds to args.roi_big (coarse scale k_A)",
    )
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=32,
        help="Number of large views generated per sample (corresponds to `num` in VoCoAugmentation)",
    )

    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--epochs", default=10000, type=int, help="number of training epochs")
    parser.add_argument(
        "--num_steps", default=300000, type=int, help="number of training iterations"
    )
    parser.add_argument("--eval_num", default=500, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="warmup steps")

    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=1024, type=int, help="embedding size")

    parser.add_argument(
        "--batch_size", default=4, type=int, help="batch size"
    )
    parser.add_argument("--lr", default=2.5e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--lrdecay", default=True, help="enable learning rate decay"
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="maximum gradient norm"
    )

    # Inside each scale: L^s = L_pred^s + lambda_reg * L_reg^s
    parser.add_argument(
        "--lambda_reg",
        default=1.0,
        type=float,
        help="Weight λ to balance L_pred and L_reg inside each scale (default: 1.0)",
    )
    # Across scales: L_total = L^A + beta_dual * L^B
    parser.add_argument(
        "--beta_dual",
        default=0.7,
        type=float,
        help="Dual-scale weight β for L_total = L^A + β·L^B",
    )
    # If you only use a single scale for now (e.g., original 4×4 grid),
    # you can set these to 0 and the framework will degenerate to single-scale mode.
    parser.add_argument(
        "--num_blocks_A",
        default=64,
        type=int,
        help="Number of sub-blocks at scale A (e.g., 4×4×4 = 64)",
    )
    parser.add_argument(
        "--num_blocks_B",
        default=512,
        type=int,
        help="Number of sub-blocks at scale B (e.g., 8×8×8 = 512)",
    )

    parser.add_argument(
        "--opt", default="adamw", type=str, help="optimization algorithm"
    )
    parser.add_argument(
        "--lr_schedule", default="warmup_cosine", type=str
    )
    parser.add_argument(
        "--resume", default=None, type=str, help="path to a checkpoint to resume training from"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument(
        "--grad_clip", action="store_true", help="enable gradient clipping"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        help="url used to set up distributed training",
    )

    args = parser.parse_args()
    logdir = args.logdir

    ngpu = torch.cuda.device_count()
    print(f"Found {ngpu} GPU(s) available.")

    args.amp = False
    torch.backends.cudnn.benchmark = True
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.dist_url
        )
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. "
            "Process %d, total %d." % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPU.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    model = Head(args)
    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "\n▶ Model has {:,} trainable parameters ({:.2f} Million).\n".format(
            total_params, total_params / 1e6
        )
    )

    if ngpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {ngpu} GPUs")
    else:
        print("Using single GPU")

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.opt}")

    global_step = 0
    if args.resume:
        print("Resume from previous checkpoints")
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict, strict=False)
        global_step = model_dict.get("global_step", 0)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps
            )
        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        else:
            scheduler = None
    else:
        scheduler = None

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    dataset = seis_dataset(args, shape=(128, 128, 128), pretrain=True)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        num_workers=0,
        pin_memory=True,
    )

    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)

    checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(logdir, "final_model.pth"))
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), os.path.join(logdir, "final_model.pth"))
    save_ckp(checkpoint, os.path.join(logdir, "model_final_epoch.pt"))


if __name__ == "__main__":
    main()
