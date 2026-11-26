import unittest

import torch

from grid_evo.policy import evaluate, fit, new_policy, training_data
from grid_evo.world import Board, FOOD, distances, expert_examples, make_splits


class WorldTests(unittest.TestCase):
    def test_split_isolation(self):
        splits = make_splits(19, 12, 6, 4)
        self.assertEqual(splits, make_splits(19, 12, 6, 4))
        self.assertEqual([len(split) for split in splits], [12, 6, 4])
        boards = [board for split in splits for board in split]
        self.assertEqual(len({board.walls for board in boards}), len(boards))

    def test_expert_takes_a_shortest_legal_step_for_each_food_subset(self):
        train, _, _ = make_splits(19, 4, 2, 2)
        for board in train:
            for observation, position, action in expert_examples([board]):
                food = [index for index, cell in enumerate(observation) if cell == FOOD]
                distance = distances(board.walls, food)
                after = board.move(position, action)
                self.assertEqual(distance[after], distance[position] - 1)

    def test_rollout_consumes_food_and_stops_when_it_hits_a_wall(self):
        class East(torch.nn.Module):
            def forward(self, observations, positions):
                return torch.tensor([[0.0, 1.0, 0.0, 0.0]]).expand(len(positions), -1)

        boards = [
            Board(walls=frozenset({4}), food=(1, 2, 3), start=0),
            Board(walls=frozenset({2}), food=(1, 5, 6), start=0),
        ]
        self.assertEqual(evaluate(East(), boards, "cpu"), {
            "food_collected": 4, "food_available": 6, "worlds_cleared": 1, "worlds": 2,
        })

    def test_backprop_updates_transformer_parameters(self):
        torch.set_num_threads(1)
        train, _, _ = make_splits(19, 4, 2, 2)
        model = new_policy(1)
        before = model.encoder.layers[0].self_attn.in_proj_weight.detach().clone()
        fit(model, training_data(train), steps=2, seed=2, device="cpu")
        self.assertFalse(torch.equal(before, model.encoder.layers[0].self_attn.in_proj_weight))
