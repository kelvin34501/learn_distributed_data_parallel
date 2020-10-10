import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data as torch_data
from termcolor import cprint

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


# test dataset
class ExampleDataset(torch_data.Dataset):
    def __init__(self, split):
        super().__init__()
        if split == "train":
            self.len = 2333
        else:
            self.len = 666

    def __getitem__(self, index):
        return torch.randn(10), torch.randn(5)

    def __len__(self):
        return self.len


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

    # create dataset & dataloader
    train_dataset = ExampleDataset("train")
    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=20, shuffle=True, num_workers=4, drop_last=True
    )

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch_id in range(100):
        if rank == 0:
            cprint(f"train epoch id: {epoch_id}", "blue")

        if rank == 0:
            pbar = tqdm(total=len(train_loader))

        for iter_id, data_batch in enumerate(train_loader):
            inp, lbl = data_batch
            inp, lbl = inp.to(rank), lbl.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inp)
            loss = loss_fn(outputs, lbl).backward()
            optimizer.step()

            if rank == 0:
                pbar.update()

        if rank == 0:
            pbar.close()

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
