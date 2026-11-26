"""Run one training process per device and select for held-out food collection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import errno
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
from random import Random
import tempfile
import time

import torch

from .policy import (
    BATCH_SIZE, FEEDFORWARD, HEADS, LAYERS, LEARNING_RATE, MUTATION_STD, WIDTH,
    evaluate, fit, mutate, new_policy, training_data,
)
from .world import PELLETS, SIZE, STEP_LIMIT, make_splits


@dataclass(frozen=True)
class Candidate:
    name: str
    founder_seed: int
    parent: str | None = None
    mutation_seed: int | None = None


def initialize_worker(devices, train, validation):
    global worker_device, worker_data, worker_validation
    worker_device = devices.get()
    torch.set_num_threads(1)
    if worker_device.startswith("cuda:"):
        torch.cuda.set_device(worker_device)
    worker_data = training_data(train)
    worker_validation = validation


def run_candidate(job):
    candidate, scratch, steps, batch_seed, epoch = job
    started = time.monotonic()
    model = new_policy(candidate.founder_seed)
    if candidate.parent is not None:
        model.load_state_dict(torch.load(
            scratch / f"{candidate.parent}.pt", map_location="cpu", weights_only=True
        ))
    if candidate.mutation_seed is not None:
        mutate(model, candidate.mutation_seed)
    model.to(worker_device)
    loss = fit(model, worker_data, steps, batch_seed, worker_device)
    validation = evaluate(model, worker_validation, worker_device)
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save(state, scratch / f"{candidate.name}.pt")
    return {
        "candidate": asdict(candidate),
        "device": worker_device,
        "worker_pid": os.getpid(),
        "started_seconds": round(started - epoch, 6),
        "finished_seconds": round(time.monotonic() - epoch, 6),
        "mean_training_loss": loss,
        "validation": validation,
    }


def generation_results(pool, jobs):
    pending = [pool.submit(run_candidate, job) for job in jobs]
    for job, future in zip(jobs, pending):
        attempts = 1
        try:
            result = future.result()
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT):
                raise
            print(f"Retrying {job[0].name} after a transient I/O error", flush=True)
            result = pool.submit(run_candidate, job).result()
            attempts = 2
        yield {**result, "attempts": attempts}


def rank(result):
    scores = result["validation"]
    return -scores["food_collected"], -scores["worlds_cleared"], result["candidate"]["name"]


def peak_concurrency(generations):
    events = []
    for generation in generations:
        for job in generation["results"]:
            events.extend(((job["started_seconds"], 1), (job["finished_seconds"], -1)))
    running = peak = 0
    for _, change in sorted(events):
        running += change
        peak = max(peak, running)
    return peak


def evolve(args, splits, devices, scratch):
    train, validation, test = splits
    rng = Random(args.seed)
    context = mp.get_context("spawn")
    generations = []
    all_winners = []
    device_queue = context.Queue()
    for device in devices:
        device_queue.put(device)
    epoch = time.monotonic()
    try:
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=context,
            initializer=initialize_worker, initargs=(device_queue, train, validation),
        ) as pool:
            parent = None
            for generation in range(args.generations):
                batch_seed = rng.getrandbits(63)
                candidates = []
                for index in range(args.population):
                    name = f"g{generation:03d}-{index:03d}"
                    if parent is None:
                        candidate = Candidate(name, rng.getrandbits(63))
                    else:
                        mutation_seed = None if index == 0 else rng.getrandbits(63)
                        candidate = Candidate(name, parent["founder_seed"], parent["name"], mutation_seed)
                    candidates.append(candidate)
                jobs = [(candidate, scratch, args.train_steps, batch_seed, epoch) for candidate in candidates]
                results = list(generation_results(pool, jobs))
                winner = min(results, key=rank)
                parent = winner["candidate"]
                all_winners.append(winner)
                generations.append({
                    "generation": generation, "batch_seed": batch_seed,
                    "winner": parent["name"], "results": results,
                })
                score = winner["validation"]
                print(
                    f"Generation {generation} | {parent['name']} | validation food "
                    f"{score['food_collected']}/{score['food_available']} | "
                    f"cleared {score['worlds_cleared']}/{score['worlds']}", flush=True,
                )
    finally:
        device_queue.close()
        device_queue.join_thread()

    best = min(all_winners, key=rank)
    selected = best["candidate"]
    torch.set_num_threads(1)
    model = new_policy(selected["founder_seed"])
    before = evaluate(model, test, "cpu")
    model.load_state_dict(torch.load(
        scratch / f"{selected['name']}.pt", map_location="cpu", weights_only=True
    ))
    cpu_validation = evaluate(model, validation, "cpu")
    after = evaluate(model, test, "cpu")
    return {
        "settings": {
            "seed": args.seed, "population": args.population,
            "generations": args.generations, "train_steps_per_candidate": args.train_steps,
            "train_worlds": len(train), "validation_worlds": len(validation), "test_worlds": len(test),
            "devices": devices, "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "grid_size": SIZE, "food_per_board": PELLETS, "rollout_step_limit": STEP_LIMIT,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "width": WIDTH, "heads": HEADS, "layers": LAYERS, "feedforward": FEEDFORWARD,
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "mutation_std": MUTATION_STD,
        },
        "peak_parallel_jobs": peak_concurrency(generations),
        "generations": generations,
        "selected_candidate": selected["name"],
        "validation_recheck": {
            "worker_device": best["device"], "worker": best["validation"],
            "cpu": cpu_validation, "same_scores": cpu_validation == best["validation"],
        },
        "test": {"device": "cpu", "untrained_founder": before, "selected_policy": after},
    }


def requested_devices(names):
    if names is None:
        count = torch.cuda.device_count()
        return [f"cuda:{index}" for index in range(count)] or ["cpu", "cpu"]
    gpu_names = [name for name in names if name != "cpu"]
    if len(set(gpu_names)) != len(gpu_names):
        raise ValueError("Each CUDA device can have only one worker")
    if gpu_names:
        available = {f"cuda:{index}" for index in range(torch.cuda.device_count())}
        if not available:
            raise ValueError("No CUDA devices detected. Use --devices cpu cpu for a local run.")
        if any(name not in available for name in gpu_names):
            raise ValueError(f"CUDA devices must be in {sorted(available)} (or use cpu)")
    return names


def positive_int(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(prog="python -m grid_evo", description=__doc__)
    parser.add_argument("--devices", nargs="+", help="for example cpu cpu or cuda:0 cuda:1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--population", type=positive_int, default=4)
    parser.add_argument("--generations", type=positive_int, default=3)
    parser.add_argument("--train-steps", type=positive_int, default=140)
    parser.add_argument("--train-worlds", type=positive_int, default=96)
    parser.add_argument("--validation-worlds", type=positive_int, default=48)
    parser.add_argument("--test-worlds", type=positive_int, default=64)
    parser.add_argument("--output", type=Path, default=Path("runs/demo.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists. Choose a new path: {args.output}")
    try:
        devices = requested_devices(args.devices)
        splits = make_splits(args.seed, args.train_worlds, args.validation_worlds, args.test_worlds)
    except ValueError as error:
        parser.error(str(error))

    print(f"Workers: {', '.join(devices)} | population: {args.population}", flush=True)
    with tempfile.TemporaryDirectory(prefix="gridworld-evo-") as scratch:
        result = evolve(args, splits, devices, Path(scratch))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    test = result["test"]
    baseline, trained = test["untrained_founder"], test["selected_policy"]
    print(
        f"Test food: {baseline['food_collected']} -> {trained['food_collected']}"
        f"/{trained['food_available']} | peak concurrency: {result['peak_parallel_jobs']} "
        f"| report: {args.output}"
    )
