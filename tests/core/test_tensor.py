import unittest
from nevula.core.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_init_scalar(self):
        t = Tensor(5.0)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.strides, ())
        self.assertEqual(t.data, [5.0])
        self.assertEqual(t.offset, 0)
        self.assertEqual(t[()], 5.0)
        self.assertEqual(t.to_list(), 5.0)

    def test_init_1d(self):
        t = Tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.strides, (1,))
        self.assertEqual(t.data, [1.0, 2.0, 3.0])
        self.assertEqual(t[0], 1.0)
        self.assertEqual(t[2], 3.0)
        self.assertEqual(t.to_list(), [1.0, 2.0, 3.0])

    def test_init_2d_inferred(self):
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.strides, (3, 1))
        self.assertEqual(t.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(t[0, 0], 1.0)
        self.assertEqual(t[1, 2], 6.0)
        self.assertEqual(t.to_list(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_indexing_bounds_and_negatives(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t[0, 0], 1)
        self.assertEqual(t[1, 1], 4)
        self.assertEqual(t[-1, -1], 4)
        self.assertEqual(t[-2, 0], 1)
        
        with self.assertRaises(IndexError):
            _ = t[2, 0]
        with self.assertRaises(IndexError):
            _ = t[0, -3]

    def test_partial_indexing_views(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        row1 = t[0]
        self.assertTrue(isinstance(row1, Tensor))
        self.assertEqual(row1.shape, (3,))
        self.assertEqual(row1.strides, (1,))
        self.assertEqual(row1.offset, 0)
        self.assertEqual(row1.to_list(), [1, 2, 3])

        row2 = t[1]
        self.assertEqual(row2.shape, (3,))
        self.assertEqual(row2.offset, 3)
        self.assertEqual(row2[1], 5)
        self.assertEqual(row2.to_list(), [4, 5, 6])

    def test_setitem_element_and_subtensor(self):
        t = Tensor([[1, 2], [3, 4]])
        t[0, 1] = 10
        self.assertEqual(t.to_list(), [[1, 10], [3, 4]])

        # Sub-tensor setitem with scalar broadcast
        t[1] = 9
        self.assertEqual(t.to_list(), [[1, 10], [9, 9]])

        # Sub-tensor setitem with list
        t[0] = [7, 8]
        self.assertEqual(t.to_list(), [[7, 8], [9, 9]])

        # Sub-tensor setitem with Tensor
        t[0] = Tensor([5, 6])
        self.assertEqual(t.to_list(), [[5, 6], [9, 9]])

    def test_reshape(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        t_reshaped = t.reshape((3, 2))
        self.assertEqual(t_reshaped.shape, (3, 2))
        self.assertEqual(t_reshaped.strides, (2, 1))
        self.assertEqual(t_reshaped.to_list(), [[1, 2], [3, 4], [5, 6]])

        # -1 inference
        t_inferred = t.reshape((-1, 2))
        self.assertEqual(t_inferred.shape, (3, 2))

        # Reshape non-contiguous view (transpose then reshape)
        t_t = t.transpose(0, 1)
        self.assertFalse(t_t.is_contiguous())
        t_t_reshaped = t_t.reshape((6,))
        self.assertEqual(t_t_reshaped.to_list(), [1, 4, 2, 5, 3, 6])

    def test_transpose(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        t_t = t.transpose(0, 1)
        self.assertEqual(t_t.shape, (3, 2))
        self.assertEqual(t_t.strides, (1, 3))
        self.assertEqual(t_t.to_list(), [[1, 4], [2, 5], [3, 6]])

        # Modifying original changes view elements
        t[0, 1] = 99
        self.assertEqual(t_t[1, 0], 99)

    def test_arithmetic_broadcasting(self):
        # Addition same shape
        self.assertEqual((Tensor([1, 2]) + Tensor([3, 4])).to_list(), [4, 6])
        # Multiplication same shape
        self.assertEqual((Tensor([1, 2]) * Tensor([3, 4])).to_list(), [3, 8])

        # Addition scalar
        self.assertEqual((Tensor([1, 2]) + 5).to_list(), [6, 7])
        # Right addition scalar
        self.assertEqual((5 + Tensor([1, 2])).to_list(), [6, 7])

        # Multiplication scalar
        self.assertEqual((Tensor([1, 2]) * 3).to_list(), [3, 6])
        # Right multiplication scalar
        self.assertEqual((3 * Tensor([1, 2])).to_list(), [3, 6])

        # Complex broadcasting
        a = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
        b = Tensor([10, 20])          # shape (2,)
        c = a + b
        self.assertEqual(c.to_list(), [[11, 22], [13, 24]])

        d = Tensor([[10], [20]])      # shape (2, 1)
        e = a + d
        self.assertEqual(e.to_list(), [[11, 12], [23, 24]])


if __name__ == '__main__':
    unittest.main()
