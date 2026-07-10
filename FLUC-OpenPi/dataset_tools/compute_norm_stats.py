"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class ExtractStateActions(transforms.DataTransformFn):
    """Keep only the fields needed for norm stats.

    This avoids image-dependent transforms for LeRobot datasets that already expose
    raw `state` and chunked `actions` columns.
    """

    def __call__(self, x: dict) -> dict:
        return {
            "state": x["state"],
            "actions": x["actions"],
        }


class LightweightLeRobotNormDataset:
    """Minimal LeRobot dataset for norm stats without image/video decode.

    This bypasses `LeRobotDataset.__init__`, which is where the expensive
    shard loading + timestamp sync check happens before our old fast path could run.
    """

    def __init__(self, repo_id: str, action_horizon: int):
        self._meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        self._episode_data_index = lerobot_dataset.get_episode_data_index(self._meta.episodes, None)

        delta_timestamps = {"actions": [t / self._meta.fps for t in range(action_horizon)]}
        lerobot_dataset.check_delta_timestamps(delta_timestamps, self._meta.fps, 1e-4)
        self._delta_indices = lerobot_dataset.get_delta_indices(delta_timestamps, self._meta.fps)

        hf_dataset = lerobot_dataset.load_dataset("parquet", data_dir=str(self._meta.root / "data"), split="train")
        keep_columns = {"state", "actions", "episode_index"}
        remove_columns = [col for col in hf_dataset.column_names if col not in keep_columns]
        if remove_columns:
            hf_dataset = hf_dataset.remove_columns(remove_columns)
        hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)
        self._hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def _get_action_indices(self, idx: int, ep_idx: int) -> list[int]:
        ep_start = self._episode_data_index["from"][ep_idx].item()
        ep_end = self._episode_data_index["to"][ep_idx].item()
        return [
            max(ep_start, min(ep_end - 1, idx + delta))
            for delta in self._delta_indices["actions"]
        ]

    def __getitem__(self, idx: int) -> dict:
        item = self._hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        action_indices = self._get_action_indices(idx, ep_idx)
        action_batch = self._hf_dataset[action_indices]["actions"]
        actions = torch.stack(action_batch)
        return {
            "state": item["state"],
            "actions": actions,
        }


def _try_make_visual_free_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
) -> _data_loader.Dataset | None:
    if data_config.repo_id is None:
        return None
    if data_config.repo_id == "fake":
        return None

    if data_config.action_sequence_keys != ("actions",):
        return None

    # This fast path is only safe for Libero-style configs where the main input
    # transform just repacks image/state/prompt and leaves state/actions untouched,
    # optionally followed by a DeltaActions transform.
    supported_passthrough = {"ActiveFlyInputs", "LiberoInputs"}
    extra_transforms: list[transforms.DataTransformFn] = []
    for transform in data_config.data_transforms.inputs:
        transform_name = type(transform).__name__
        if isinstance(transform, transforms.DeltaActions):
            extra_transforms.append(transform)
            continue
        if transform_name in supported_passthrough:
            continue
        return None

    try:
        dataset = LightweightLeRobotNormDataset(data_config.repo_id, action_horizon=action_horizon)
    except Exception as exc:
        print(f"Lightweight norm-stats path unavailable, falling back to default loader: {exc}", flush=True)
        return None

    return _data_loader.TransformedDataset(
        dataset,
        [
            *extra_transforms,
            RemoveStrings(),
        ],
    )


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    fast_dataset = _try_make_visual_free_dataset(data_config, action_horizon)
    if fast_dataset is not None:
        print("Using lightweight visual-free norm-stats path: skipping LeRobot image decode/checks.", flush=True)
        dataset = fast_dataset
    else:
        dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
                RemoveStrings(),
            ],
        )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_for_config(
    config: _config.TrainConfig,
    max_frames: int | None = None,
    num_workers: int | None = None,
) -> Path:
    data_config = config.data.create(config.assets_dirs, config.model)
    effective_num_workers = config.num_workers if num_workers is None else num_workers

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            effective_num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = data_config.asset_id or data_config.repo_id
    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return Path(output_path) / "norm_stats.json"


def main(config_name: str, max_frames: int | None = None, num_workers: int | None = None):
    config = _config.get_config(config_name)
    compute_for_config(config, max_frames=max_frames, num_workers=num_workers)


if __name__ == "__main__":
    tyro.cli(main)
