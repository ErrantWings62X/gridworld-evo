"""A standard Transformer encoder trained with supervised backpropagation."""

import torch
from torch import nn

from .world import CELLS, PELLETS, SIZE, STEP_LIMIT, expert_examples

WIDTH = 32
HEADS = 4
LAYERS = 2
FEEDFORWARD = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.002
MUTATION_STD = 0.015


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(4, WIDTH)
        self.position = nn.Linear(2, WIDTH, bias=False)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=WIDTH,
                nhead=HEADS,
                dim_feedforward=FEEDFORWARD,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=LAYERS,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 4))
        cells = torch.arange(CELLS)
        self.register_buffer(
            "coordinates", torch.stack((cells // SIZE, cells % SIZE), dim=1).float(),
            persistent=False,
        )

    def forward(self, observations, positions):
        relative = self.coordinates - self.coordinates[positions, None]
        tokens = self.tokens(observations) + self.position(relative / (SIZE - 1))
        features = self.encoder(tokens)
        rows = torch.arange(len(positions), device=positions.device)
        return self.head(features[rows, positions])


def new_policy(seed):
    # Initialization and mutation happen on CPU, even for CUDA jobs.
    torch.random.default_generator.manual_seed(seed)
    return Policy()


def mutate(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(noise, alpha=MUTATION_STD)


def training_data(boards):
    observations, positions, actions = zip(*expert_examples(boards))
    return tuple(torch.tensor(values, dtype=torch.long) for values in (
        observations, positions, actions
    ))


def fit(model, data, steps, seed, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    loss_sum = torch.zeros((), device=device)
    for _ in range(steps):
        indices = torch.randint(len(data[0]), (BATCH_SIZE,), generator=generator)
        observations, positions, actions = (tensor[indices].to(device) for tensor in data)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(observations, positions), actions)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach()
    return loss_sum.item() / steps


@torch.inference_mode()
def evaluate(model, boards, device):
    """Roll out greedy actions and score actual food collected."""
    model.eval()
    positions = [board.start for board in boards]
    food = [frozenset(board.food) for board in boards]
    seen = [set() for _ in boards]
    active = list(range(len(boards)))
    for _ in range(STEP_LIMIT):
        if not active:
            break
        observations = [boards[i].observe(positions[i], food[i]) for i in active]
        actions = model(
            torch.tensor(observations, device=device),
            torch.tensor([positions[i] for i in active], device=device),
        ).argmax(dim=1).tolist()
        next_active = []
        for i, action in zip(active, actions):
            seen[i].add((positions[i], food[i]))
            positions[i] = boards[i].move(positions[i], action)
            food[i] = food[i] - {positions[i]}
            if food[i] and (positions[i], food[i]) not in seen[i]:
                next_active.append(i)
        active = next_active
    return {
        "food_collected": sum(PELLETS - len(remaining) for remaining in food),
        "food_available": PELLETS * len(boards),
        "worlds_cleared": sum(not remaining for remaining in food),
        "worlds": len(boards),
    }
