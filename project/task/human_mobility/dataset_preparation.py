import os.path
import hydra
import numpy as np
import pandas as pd
import torch
from flwr.common.logger import log
import logging
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import requests
import zipfile

from project.types.common import HumanMobilityDataset

DOWNLOAD_URL = "http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
FEATURES = ["latitude", "longitude"]
LABEL = "check_ins"


def _download_data(cfg):
    dataset_dir = Path(cfg.dataset.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = DOWNLOAD_URL.split("/")[-1]
    download_path = dataset_dir / filename

    if not os.path.exists(download_path):
        print(f"Downloading {filename}...")
        response = requests.get(DOWNLOAD_URL)
        response.raise_for_status()

        with open(download_path, "wb") as file:
            file.write(response.content)
        print("Download complete.")

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")

    print("Processing train/test split...")
    filename_no_ext = filename.split(".")[0]
    trainset, testset = _load_data(
        dataset_dir=dataset_dir / filename_no_ext / cfg.dataset.data_file,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.dataset.seed,
        pred_hours=cfg.dataset.pred_hours,
    )
    print("Processing complete.")
    return trainset, testset


def _transform_per_venue(df, window="48H"):
    agg_operations = {feature: "first" for feature in FEATURES}
    feat_info = df.groupby("venue_id").agg(agg_operations).reset_index()

    df.set_index("local_time", inplace=True)

    def resample_group(group):
        resampled = group.resample(window).size().reset_index(name="check_ins")
        resampled["venue_id"] = group.name
        return resampled

    resampled_check_ins = (
        df.groupby("venue_id", group_keys=False)
        .apply(resample_group)
        .reset_index(drop=True)
    )

    return pd.merge(resampled_check_ins, feat_info, on="venue_id", how="left")


def _load_data(dataset_dir, val_ratio, seed, pred_hours, encoding="ISO-8859-1"):
    headers = [
        "user_id",
        "venue_id",
        "venue_category_id",
        "venue_category_name",
        "latitude",
        "longitude",
        "timezone_offset",
        "utc_time",
    ]
    df = pd.read_csv(
        dataset_dir, sep="\t", header=None, names=headers, encoding=encoding
    )

    # Convert "utc_time" to datetime and adjust to local time
    df["utc_time"] = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y")
    df["local_time"] = df["utc_time"] + pd.to_timedelta(df["timezone_offset"], unit="m")

    # Transform into time slot windows per venue
    df = _transform_per_venue(df, window=f"{pred_hours}H")

    # Add cyclical daily features
    df["day_sin"] = np.sin(df["local_time"].dt.hour * (2.0 * np.pi / 24))
    df["day_cos"] = np.cos(df["local_time"].dt.hour * (2.0 * np.pi / 24))

    # Add cyclical yearly features
    df["year_day_sin"] = np.sin(df["local_time"].dt.dayofyear * (2.0 * np.pi / 365.25))
    df["year_day_cos"] = np.cos(df["local_time"].dt.dayofyear * (2.0 * np.pi / 365.25))

    FEATURES.extend(["day_sin", "day_cos", "year_day_sin", "year_day_cos"])

    # Aggregate number of check-ins per venue on the last `pred_hours`
    """
    df["time_slot"] = df["local_time"].dt.floor(f"{pred_hours}H")
    df_with_checkins = df.groupby(["venue_id", "time_slot"] + FEATURES).size().reset_index(name=LABEL)
    grouped_by_venues = df_with_checkins.groupby("venue_id")
    """

    # Split the grouped venues dataframe into train and test
    groups = [group for _, group in df.groupby("venue_id")]
    train_groups, test_groups = train_test_split(
        groups, test_size=val_ratio, random_state=seed
    )

    train_df = pd.concat(train_groups).reset_index(drop=True)
    test_df = pd.concat(test_groups).reset_index(drop=True)

    return train_df, test_df


def _partition_data(train_df, test_df, num_clients, seed, iid, power_law, balance):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Partitioning data into {num_clients} clients...")

    # Calculate the number of sequences for each venue
    venue_sequence_counts = train_df["venue_id"].value_counts()

    # Sort venues by the number of sequences, descending, and select the top `num_clients`
    top_venue_ids = venue_sequence_counts.nlargest(num_clients).index

    random_venue_ids = np.random.choice(train_df["venue_id"].unique(), num_clients)

    datasets = []
    for venue_id in top_venue_ids:
        client_df = train_df[train_df["venue_id"] == venue_id]

        # Standardise client features.
        # Note: scaler is re-fitted for each client, mimicking a realistic scenario in which venues
        # cannot share their internal dataset distributions.
        # scaler = StandardScaler()
        # client_df[FEATURES] = scaler.fit_transform(client_df[FEATURES])
        client_ds = HumanMobilityDataset(
            client_df,
            sequence_length=16,
            prediction_length=1,
            features=FEATURES,
            label=LABEL,
        )

        datasets.append(client_ds)

    # Standardise test features
    # scaler = StandardScaler()
    # test_df[FEATURES] = scaler.fit_transform(test_df[FEATURES])

    test_venue_counts = test_df["venue_id"].value_counts()
    test_top_venues = test_venue_counts.nlargest(100).index
    fed_test_set = HumanMobilityDataset(
        test_df[test_df["venue_id"].isin(test_top_venues)],
        sequence_length=16,
        prediction_length=1,
        features=FEATURES,
        label=LABEL,
    )

    print("All done.")

    return datasets, fed_test_set


@hydra.main(
    config_path="../../conf",
    config_name="human_mobility",
    version_base=None,
)
def download_and_preprocess(cfg):
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    train_df, test_df = _download_data(cfg)

    client_datasets, fed_test_set = _partition_data(
        train_df,
        test_df,
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

        len_val = int(len(client_dataset) / (1 / cfg.dataset.val_ratio))
        lengths = [len(client_dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            client_dataset,
            lengths,
            torch.Generator().manual_seed(cfg.dataset.seed),
        )
        torch.save(ds_train, client_dir / "train.pt")
        torch.save(ds_val, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
