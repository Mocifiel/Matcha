import argparse
import logging
import math
import os
import sys

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from six.moves import urllib
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))
from dist import ompi_rank, ompi_size, ompi_local_rank, dist_init  # noqa: E402
from parallel.data_parallel import BalancedDataParallel  # noqa: E402

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


# Top-program should be responsible for setting logging behaviors.
LOG = logging.getLogger()
LOG_FORMAT = logging.Formatter("[%(asctime)s %(name)-5s %(levelname)s P%(process)d]$ %(message)s")
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
STREAM_HANDLER.setFormatter(LOG_FORMAT)
LOG.addHandler(STREAM_HANDLER)
LOG.setLevel(logging.INFO)


def synchronize(world_size):
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if world_size == 1:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, rank):
    model.train()
    writer = SummaryWriter(args.log_dir)
    global global_step
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_step += 1
        if batch_idx % args.log_interval == 0:
            print(
                "[Rank {}] - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    rank,
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            # tensorboard
            if rank == 0:
                writer.add_scalar("loss", loss.item(), global_step)


def main():
    # Training settings

    def str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ["true", "false"]:
            raise ValueError("Argument needs to be a Boolean, got {}".format(s))
        return {"true": True, "false": False}[s.lower()]

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="model directory")
    parser.add_argument("--model-dir", type=str, required=True, help="model directory")
    parser.add_argument("--log-dir", "--logDir", type=str, default=None, help="log directory")
    parser.add_argument(
        "--distributed", type=str_to_bool, default="true", help="use distributed training or not (default: true)"
    )
    parser.add_argument(
        "--dist-method",
        type=str,
        choices=["torch"],
        default="torch",
        help="distributed method (default: torch)",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # create log directory to save tensorboard
    # summary written in the default directory can be shown on Philly
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(args.model_dir, "log")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.distributed:
        # init distributed env
        if args.dist_method == "torch":
            dist_init(backend="nccl")
            rank = ompi_rank()
            local_rank = ompi_local_rank()
            world_size = ompi_size()
            if rank == 0:
                print("[Rank 0]: DistributedDataParallel PyTorch Method")
        else:
            print("Unsupported dist method: {}".format(args.dist_method))
        torch.cuda.set_device(local_rank)
        batch_size = args.batch_size
    else:
        # init data parallel
        world_size, rank = 1, 0
        gpu_num = torch.cuda.device_count()
        num_workers = 2
        if gpu_num > 1:
            gpu0_batch_size = args.batch_size // 2
            num_workers *= math.ceil(math.log(gpu_num, 2))
        else:
            gpu0_batch_size = args.batch_size
        batch_size = args.batch_size * (gpu_num - 1) + gpu0_batch_size
        print("[Rank 0]: DataParallel: GPU = {}, batch_size = {}".format(gpu_num, batch_size))

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    user = os.environ.get(("USER"))
    user = user if user is not None else "USER"

    if rank == 0:
        # Sync master
        download = True
        train_dataset = datasets.MNIST(
            root=os.path.join(args.data_dir, user, "Data", "MNIST", "data"),
            train=True,
            download=download,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        synchronize(world_size)
    else:
        # Sync worker
        synchronize(world_size)
        download = False
        train_dataset = datasets.MNIST(
            root=os.path.join(args.data_dir, user, "Data", "MNIST", "data"),
            train=True,
            download=download,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )

    if rank == 0:
        print(args)
    # synchronize(world_size)

    # distributed dataloader
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # model wrapping
    if args.distributed:
        if args.dist_method == "torch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            print("[Rank {}]: Unsupported dist method: {}".format(rank, args.dist_method))
    else:
        model = BalancedDataParallel(gpu0_batch_size, model)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    global global_step
    global_step = 0

    run_id = os.environ.get("MLFLOW_RUN_ID", None)
    if rank == 0 and run_id is not None:
        mlflow.autolog()
        mlflow.start_run(os.environ.get("MLFLOW_RUN_ID"))
        try:
            # pytorch autologging is not enabled by default
            mlflow.pytorch.autolog(log_every_n_step=int(os.environ.get("AMLT_LOG_EVERY_N_STEP", 1)))
        except Exception:
            pass

    for epoch in range(1, args.epochs + 1):
        # if args.distributed and args.dist_method == 'torch':
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch, rank)
        scheduler.step()

    if rank == 0 and run_id is not None:
        mlflow.end_run()

    save_dir = os.path.join(args.model_dir, "mnist_cnn.pt")
    if rank == 0:
        print(f"save model to {save_dir}")
        torch.save(model.state_dict(), save_dir)


if __name__ == "__main__":
    main()
