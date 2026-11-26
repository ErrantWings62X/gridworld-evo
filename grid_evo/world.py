"""A fully observed grid with walls and three collectible food pellets."""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from random import Random

SIZE = 5
CELLS = SIZE * SIZE
PELLETS = 3
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # North, east, south, west.
EMPTY, WALL, FOOD, AGENT = range(4)
STEP_LIMIT = 32


def neighbors(cell):
    row, col = divmod(cell, SIZE)
    for dr, dc in ACTIONS:
        y, x = row + dr, col + dc
        yield y * SIZE + x if 0 <= y < SIZE and 0 <= x < SIZE else -1


NEIGHBORS = tuple(tuple(neighbors(cell)) for cell in range(CELLS))


@dataclass(frozen=True)
class Board:
    walls: frozenset[int]
    food: tuple[int, ...]
    start: int

    def observe(self, position, food):
        cells = [EMPTY] * CELLS
        for wall in self.walls:
            cells[wall] = WALL
        for pellet in food:
            cells[pellet] = FOOD
        cells[position] = AGENT
        return cells

    def move(self, position, action):
        destination = NEIGHBORS[position][action]
        return position if destination == -1 or destination in self.walls else destination


def distances(walls, targets):
    """Return shortest walk distances to the closest target."""
    result = dict.fromkeys(targets, 0)
    queue = deque(result)
    while queue:
        cell = queue.popleft()
        for near in NEIGHBORS[cell]:
            if near != -1 and near not in walls and near not in result:
                result[near] = result[cell] + 1
                queue.append(near)
    return result


def make_splits(seed, train_count, validation_count, test_count):
    """Allocate distinct wall layouts across all three splits."""
    rng = Random(seed)
    layouts = list(combinations(range(CELLS), 3))
    rng.shuffle(layouts)
    required = train_count + validation_count + test_count
    boards = []
    for layout in layouts:
        walls = frozenset(layout)
        free = [cell for cell in range(CELLS) if cell not in walls]
        if len(distances(walls, [free[0]])) != len(free):
            continue
        start, *food = rng.sample(free, 1 + PELLETS)
        boards.append(Board(walls, tuple(sorted(food)), start))
        if len(boards) == required:
            break
    if len(boards) != required:
        raise ValueError(f"Only {len(boards)} connected wall layouts are available")
    test_start = train_count + validation_count
    return boards[:train_count], boards[train_count:test_start], boards[test_start:]


def expert_examples(boards):
    """Label each player position and nonempty food subset by nearest-food BFS."""
    for board in boards:
        for count in range(1, len(board.food) + 1):
            for food in combinations(board.food, count):
                distance = distances(board.walls, food)
                for position in range(CELLS):
                    if not distance.get(position, 0):
                        continue
                    action = next(
                        action
                        for action, near in enumerate(NEIGHBORS[position])
                        if distance.get(near) == distance[position] - 1
                    )
                    yield board.observe(position, food), position, action
