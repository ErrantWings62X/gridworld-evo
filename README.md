# Gridworld Evo

Train a small Transformer to forage in a 5×5 grid, then evolve a population of policies by how much food they collect on new maps. Each worker trains an entire candidate on its assigned device. A single machine can run independent candidates in parallel on multiple CUDA GPUs or CPU processes.

## Run

The tested setup uses Python 3.12. From this directory, create a virtual environment and start a two-worker run.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m grid_evo --devices cpu cpu
```

For two NVIDIA GPUs, start with a CUDA-enabled build from the [PyTorch installer](https://pytorch.org/get-started/locally/). Then select both devices explicitly.

```sh
python -m pip install -r requirements.txt
python -m grid_evo --devices cuda:0 cuda:1 --output runs/two-gpu.json
```

Device numbers are relative to the GPUs visible to PyTorch. Omitting `--devices` uses every visible CUDA device, with two CPU workers as the fallback. The JSON report records each job's device, validation score, runtime interval, and attempt count. `peak_parallel_jobs` measures overlap. The coordinator retries a transient I/O failure once and surfaces permanent failures.

Run a short smoke test with two generations.

```sh
python -m grid_evo --devices cpu cpu --population 2 --generations 2 \
  --train-steps 3 --train-worlds 4 --validation-worlds 3 --test-worlds 4 \
  --output runs/smoke.json
```

## Experiment

1. Generate connected maps with three walls and three food pellets. Allocate distinct wall layouts to training, validation, and test.
2. Use shortest-path search to label moves toward the nearest remaining pellet. Train a two-layer Transformer encoder with cross-entropy, backpropagation, and AdamW. The policy receives the full grid and emits one of four moves.
3. Roll out each policy on validation maps and rank by pellets collected, with fully cleared maps breaking ties. The top policy supplies the next generation's starting weights. Each subsequent round trains one unmutated copy and three descendants with independently perturbed weights under the default population size.
4. Select the best validation candidate across generations and evaluate it alongside its untrained initialization on the held-out test maps. The coordinator also reruns the selected checkpoint on the validation maps on CPU and records whether its aggregate scores match the original worker's.

Greedy rollouts end after all pellets are collected, after 32 moves, or when the agent repeats a position with the same food remaining.

## Example result

Seed 17, Python 3.12.13, PyTorch 2.15.0, two CPU worker processes. The run trained 12 candidates over three generations with 140 optimizer steps per candidate. Two jobs ran concurrently. A single-worker run reproduced every candidate's validation score and the final test scores.

| Policy | Food collected | Maps cleared |
| --- | ---: | ---: |
| Untrained initialization of the selected lineage | 19 / 192 | 0 / 64 |
| Selected trained policy | 143 / 192 | 37 / 64 |

These scores measure the combined pipeline in one seeded experiment. Isolating the contribution of mutation would require a matched backprop-only baseline. The example report is from a CPU run. Training on two physical CUDA GPUs remains unverified for this demo.

See [the full run report](example-results.json) for the settings, per-candidate results, device assignments, and validation recheck.

```sh
python -m unittest discover -s tests -v
```
