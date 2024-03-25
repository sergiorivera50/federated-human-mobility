from pathlib import Path

import torch
from flwr.common.logger import log
import hydra
from torch.utils.data import Dataset, Subset, random_split, TensorDataset
from omegaconf import OmegaConf
import logging


class SyntheticRegressionDataset(Dataset):
    def __init__(self, data, labels, train: bool):
        split_idx = int(len(data) * 0.8)
        if train:
            self.data = data[:split_idx]
            self.labels = labels[:split_idx]
        else:
            self.data = data[split_idx:]
            self.labels = labels[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _download_data(dataset_dir):
    num_samples = 10000
    a, b = 2.0, 1.0  # y = ax + b

    x = torch.linspace(-10, 10, num_samples).unsqueeze(1)
    y = a * x + b + torch.randn(x.size())

    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_dataset = TensorDataset(x, y)
    torch.save(x, dataset_dir / "full.pt")

    # Split dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset


def _partition_data(
    trainset: SyntheticRegressionDataset,
    testset: SyntheticRegressionDataset,
    num_clients: int,
    seed: int,
    iid: bool,
    power_law: bool,
    balance: bool,
) -> tuple[list[Subset], SyntheticRegressionDataset]:
    torch.manual_seed(seed)  # reproduceablity

    if iid:
        # Evenly distribute data among clients for IID case
        partition_size = int(len(trainset) / num_clients)
        print(
            f"Trainset size is {len(trainset)}, while partition size is {partition_size}"
        )
        lengths = [partition_size] * num_clients
        datasets = random_split(
            trainset, lengths, generator=torch.Generator().manual_seed(seed)
        )
    else:
        # Simple non-IID partitioning: sort by labels and divide the dataset
        indices = torch.argsort(trainset.labels.squeeze())
        sorted_data = Subset(trainset, indices)
        partition_size = int(len(sorted_data) / num_clients)
        datasets = [
            Subset(sorted_data, range(i * partition_size, (i + 1) * partition_size))
            for i in range(num_clients)
        ]

    return datasets, testset


@hydra.main(
    config_path="../../conf",
    config_name="regression",
    version_base=None,
)
def download_and_preprocess(cfg):
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    trainset, testset = _download_data(
        Path(cfg.dataset.dataset_dir),
    )

    client_datasets, fed_test_set = _partition_data(
        trainset,
        testset,
        cfg.dataset.num_clients,
        cfg.dataset.seed,
        cfg.dataset.iid,
        cfg.dataset.power_law,
        cfg.dataset.balance,
    )

    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    torch.save(fed_test_set, partition_dir / "test.pt")

    for idx, client_dataset in enumerate(client_datasets):
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)

        len_val = int(
            len(client_dataset) / (1 / cfg.dataset.val_ratio),
        )
        lengths = [len(client_dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            client_dataset,
            lengths,
            torch.Generator().manual_seed(cfg.dataset.seed),
        )
        print(f"Creating train ds for client {idx} with length {len(ds_train)}")
        print(f"Creating test (val) ds for client {idx} with length {len(ds_val)}")
        torch.save(ds_train, client_dir / "train.pt")
        torch.save(ds_val, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
