import click
import time
import json
import os
import tempfile
from typing import Dict

import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def add_fake_labels(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    batch_size = len(batch["image"])
    batch["label"] = np.zeros([batch_size], dtype=int)
    return batch


def transform_image(
    batch: Dict[str, np.ndarray], transform: torch.nn.Module
) -> Dict[str, np.ndarray]:
    transformed_tensors = [transform(image).numpy() for image in batch["image"]]
    batch["image"] = transformed_tensors
    return batch


def train_loop_per_worker(config):
    raw_model = resnet18(pretrained=True)
    model = train.torch.prepare_model(raw_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_dataset_shard = train.get_dataset_shard("train")

    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        for i, data in enumerate(
            train_dataset_shard.iter_torch_batches(batch_size=config["batch_size"])
        ):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["image"].to(device=train.torch.get_device())
            labels = data["label"].to(device=train.torch.get_device())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        
        # In standard DDP training, where the model is the same across all ranks,
        # only the global rank 0 worker needs to save and report the checkpoint
        should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
        if train.get_context().get_world_rank() == 0 and should_checkpoint:    
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(model.state_dict(), os.path.join(tmpdir, "model.pt"))
                train.report(
                    dict(running_loss=running_loss),
                    checkpoint=Checkpoint.from_directory(tmpdir),
                )
        
        # # Distributed checkpointing is the best practice for saving checkpoints 
        # # when doing model-parallel training (e.g., DeepSpeed, FSDP, Megatron-LM).
        # # The checkpoint in cloud storage will contain: model-rank=0.pt, model-rank=1.pt
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     rank = train.get_context().get_world_rank()
        #     torch.save(model.state_dict(), os.path.join(tmpdir, f"model-rank={rank}.pt"))
        #     train.report(
        #         dict(running_loss=running_loss),
        #         checkpoint=Checkpoint.from_directory(tmpdir),
        #     )

@click.command(help="Run Finetuning and Batch prediction on Pytorch ResNet models.")
@click.option("--data-size-gb", type=int, default=4)
@click.option("--num-epochs", type=int, default=10)
@click.option("--num-workers", type=int, default=1)
@click.option("--gpus-per-worker", type=int, default=1)
def main(data_size_gb: int, num_epochs: int, num_workers: int, gpus_per_worker: int):
    import fsspec
    import pyarrow.fs
    from pydoopfsspec import HadoopFileSystem
    fsspec.register_implementation("pydoop", HadoopFileSystem)
    hdfs = fsspec.filesystem("pydoop")
    fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(hdfs))

    data_url = (
        "/Projects/ray_job/Resources/synthetic_images"
    )
    print(
        "Running Pytorch image model training with "
        f"{data_size_gb}GB data from {data_url}"
    )
    print(f"Training for {num_epochs} epochs with {num_workers} workers({gpus_per_worker} GPUs each).")
    start = time.time()

    print(f"Running GPU training with {data_size_gb}GB data from {data_url}")

    dataset = ray.data.read_images(data_url, size=(256, 256), filesystem=fs)

    data_loading = time.time()
    data_loading_time_s = round(data_loading - start, 4)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = dataset.map_batches(add_fake_labels)
    dataset = dataset.map_batches(transform_image, fn_kwargs={"transform": transform})

    data_preprocessing = time.time()
    data_preprocessing_time_s = round(data_preprocessing - data_loading, 4)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"batch_size": 64, "num_epochs": num_epochs},
        datasets={"train": dataset},
        scaling_config=ScalingConfig(
            num_workers=num_workers, 
            use_gpu=True,
            resources_per_worker={"GPU": gpus_per_worker},
            trainer_resources={"CPU": 0},

        ),
        run_config=RunConfig(
            storage_filesystem=fs,
            storage_path="/Projects/ray_job/Resources",
            name=f"ResNet18_benchmark",

        ),
    )
    trainer.fit()

    model_training_time_s = round(time.time() - data_preprocessing, 4)
    total_time_s = round(time.time() - start, 4)

    # For structured output integration with internal tooling
    results = {"data_size_gb": data_size_gb, "num_epochs": num_epochs}
    results["perf_metrics"] = [
        {
            "perf_metric_name": "total_time_s",
            "perf_metric_value": total_time_s,
            "perf_metric_type": "LATENCY",
        },
        {
            "perf_metric_name": "data_loading_time_s",
            "perf_metric_value": data_loading_time_s,
            "perf_metric_type": "LATENCY",
        },
        {
            "perf_metric_name": "data_preprocessing_time_s",
            "perf_metric_value": data_preprocessing_time_s,
            "perf_metric_type": "LATENCY",
        },
        {
            "perf_metric_name": "model_training_time_s",
            "perf_metric_value": model_training_time_s,
            "perf_metric_type": "LATENCY",
        },
        {
            "perf_metric_name": "image_loading_throughput_MB_s",
            "perf_metric_value": round(
                data_size_gb * 1024 / data_loading_time_s, 4
            ),
            "perf_metric_type": "THROUGHPUT",
        },
        {
            "perf_metric_name": "finetuning_speed_seconds_epochs",
            "perf_metric_value": round(
                model_training_time_s / num_epochs, 4
            ),
            "perf_metric_type": "COMPUTATION",
        },
    ]

    test_output_json = os.environ.get("TEST_OUTPUT_JSON", "/tmp/release_test_out.json")
    with open(test_output_json, "wt") as f:
        json.dump(results, f)

    print(results)


if __name__ == "__main__":
    main()