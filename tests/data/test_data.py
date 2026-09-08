import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nevula.core.tensor import Tensor
from nevula.nn.linear import Linear
from nevula.nn.losses import MSELoss
from nevula.optim.sgd import SGD
from nevula.data.dataset import Dataset, TensorDataset
from nevula.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from nevula.data.dataloader import DataLoader, default_collate
from nevula.data.transforms import Compose, ToTensor, Normalize, Lambda


class TestData(unittest.TestCase):

    def test_custom_dataset(self):
        """Custom dataset subclass implementing __len__ and __getitem__"""
        class SimpleDataset(Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return idx * 2

        ds = SimpleDataset(5)
        self.assertEqual(len(ds), 5)
        self.assertEqual(ds[2], 4)

    def test_tensor_dataset(self):
        """TensorDataset wraps multiple tensors with matching first dimension"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = Tensor([[10.0], [20.0], [30.0]])

        dataset = TensorDataset(x, y)
        self.assertEqual(len(dataset), 3)

        sample_x, sample_y = dataset[1]
        self.assertEqual(sample_x.to_list(), [3.0, 4.0])
        self.assertEqual(sample_y.to_list(), [20.0])

        # Mismatched length raises error
        mismatched_y = Tensor([[10.0], [20.0]])
        with self.assertRaises(ValueError):
            _ = TensorDataset(x, mismatched_y)

    def test_samplers(self):
        """SequentialSampler, RandomSampler, and BatchSampler"""
        data = list(range(10))

        # SequentialSampler
        seq_sampler = SequentialSampler(data)
        self.assertEqual(list(seq_sampler), list(range(10)))
        self.assertEqual(len(seq_sampler), 10)

        # RandomSampler
        rand_sampler = RandomSampler(data)
        rand_indices = list(rand_sampler)
        self.assertEqual(sorted(rand_indices), list(range(10)))
        self.assertEqual(len(rand_sampler), 10)

        # BatchSampler without drop_last
        batch_sampler = BatchSampler(seq_sampler, batch_size=4, drop_last=False)
        batches = list(batch_sampler)
        self.assertEqual(batches, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual(len(batch_sampler), 3)

        # BatchSampler with drop_last
        batch_sampler_drop = BatchSampler(seq_sampler, batch_size=4, drop_last=True)
        batches_drop = list(batch_sampler_drop)
        self.assertEqual(batches_drop, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(len(batch_sampler_drop), 2)

    def test_default_collate(self):
        """default_collate stacks tensors into mini-batches"""
        # Batch of 1D tensors -> 2D batch
        tensors = [Tensor([1.0, 2.0]), Tensor([3.0, 4.0]), Tensor([5.0, 6.0])]
        collated = default_collate(tensors)
        self.assertEqual(collated.shape, (3, 2))
        self.assertEqual(collated.to_list(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Batch of (x, y) tuples
        tuple_batch = [
            (Tensor([1.0, 2.0]), Tensor([10.0])),
            (Tensor([3.0, 4.0]), Tensor([20.0])),
        ]
        bx, by = default_collate(tuple_batch)
        self.assertEqual(bx.shape, (2, 2))
        self.assertEqual(by.shape, (2, 1))

    def test_dataloader_iteration(self):
        """DataLoader batches and yields samples correctly"""
        x_data = [[float(i), float(i * 2)] for i in range(10)]
        y_data = [[float(i * 10)] for i in range(10)]
        dataset = TensorDataset(Tensor(x_data), Tensor(y_data))

        loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
        self.assertEqual(len(loader), 3)

        batch_shapes = []
        for bx, by in loader:
            batch_shapes.append((bx.shape, by.shape))

        expected = [
            ((4, 2), (4, 1)),
            ((4, 2), (4, 1)),
            ((2, 2), (2, 1)),
        ]
        self.assertEqual(batch_shapes, expected)

    def test_transforms(self):
        """Compose, ToTensor, Normalize, and Lambda transforms"""
        pipeline = Compose([
            ToTensor(),
            Normalize(mean=10.0, std=2.0),
            Lambda(lambda t: t + 1.0),
        ])

        raw_data = [10.0, 14.0]
        out = pipeline(raw_data)
        # (10 - 10) / 2 + 1 = 1.0
        # (14 - 10) / 2 + 1 = 3.0
        self.assertEqual(out.to_list(), [1.0, 3.0])

    def test_end_to_end_dataloader_training(self):
        """The Phase 5 Finish Line: training a neural network using DataLoader"""
        # 16 samples, 2 features, target = 2 * x1 + 3 * x2
        samples_x = [[float(i), float(i + 1)] for i in range(16)]
        samples_y = [[2.0 * x1 + 3.0 * x2] for x1, x2 in samples_x]

        dataset = TensorDataset(Tensor(samples_x), Tensor(samples_y))
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = Linear(2, 1)
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.005)

        # Train for 20 epochs across mini-batches
        initial_loss = None
        for epoch in range(20):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if initial_loss is None:
                    initial_loss = loss.item()
                loss.backward()
                optimizer.step()

        # Training completes with updated parameters
        self.assertIsNotNone(initial_loss)
        self.assertTrue(loss.item() < initial_loss)


if __name__ == '__main__':
    unittest.main()
