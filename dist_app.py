import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from termcolor import cprint

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the progress group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    cprint(f"Running basic DDP example on rank {rank}", "green")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # one iter?
    if rank == 0:
        pbar = tqdm(total=1000)

    for iter_id in range(1000):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        label = torch.randn(20, 5).to(rank)
        loss = loss_fn(outputs, label).backward()
        optimizer.step()

        if rank == 0:
            pbar.update()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    n_gpus = torch.cuda.device_count()
    cprint(f"got {n_gpus}")
    run_demo(demo_basic, n_gpus)


# problems
# data split: How to integrate it with dataloader?
#             Deadlocks?
# eval: How to? need one method to sync outputs
# checkpoints:
