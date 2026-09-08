from typing import Any, Optional, Sequence


def _prod(shape: tuple[int, ...]) -> int:
    """Calculates the product of a shape tuple."""
    res = 1
    for x in shape:
        res *= x
    return res


def _flatten_and_infer_shape(data) -> tuple[list, tuple[int, ...]]:
    """Recursively flattens nested lists/tuples and infers their shape."""
    if not isinstance(data, (list, tuple)):
        return [data], ()
    if len(data) == 0:
        return [], (0,)

    if isinstance(data[0], (list, tuple)):
        flat_data = []
        inferred_shape = None
        for item in data:
            flat_sub, sub_shape = _flatten_and_infer_shape(item)
            if inferred_shape is None:
                inferred_shape = sub_shape
            elif inferred_shape != sub_shape:
                raise ValueError("Inconsistent shapes in nested list.")
            flat_data.extend(flat_sub)
        return flat_data, (len(data),) + inferred_shape
    else:
        if any(isinstance(item, (list, tuple)) for item in data):
            raise ValueError("Inconsistent nesting in data.")
        return list(data), (len(data),)


def broadcast_shapes(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    """Computes the broadcasted shape of two shape tuples."""
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    max_ndim = max(ndim1, ndim2)

    padded1 = (1,) * (max_ndim - ndim1) + shape1
    padded2 = (1,) * (max_ndim - ndim2) + shape2

    result_shape = []
    for d1, d2 in zip(padded1, padded2):
        if d1 == d2:
            result_shape.append(d1)
        elif d1 == 1:
            result_shape.append(d2)
        elif d2 == 1:
            result_shape.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
    return tuple(result_shape)


def _generate_indices(shape: tuple[int, ...]):
    """Generates all multi-dimensional coordinate tuples for a given shape."""
    def gen(dim):
        if dim == len(shape):
            yield ()
            return
        for i in range(shape[dim]):
            for rest in gen(dim + 1):
                yield (i,) + rest
    yield from gen(0)


class Tensor:
    """
    Multidimensional array abstraction in Nevula with support for strides,
    broadcasting, views, and reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: Any,
        shape: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        offset: int = 0,
        _clone: bool = True,
        requires_grad: bool = False,
    ):
        # 📦 1. Storage & Layout Metadata
        if isinstance(data, Tensor):
            self.data = data.data
            inferred_shape = data.shape
            inferred_strides = data.strides
            offset = data.offset
        elif not _clone and isinstance(data, list):
            self.data = data
            inferred_shape = (len(data),)
            inferred_strides = None
        else:
            flat_data, inferred_shape = _flatten_and_infer_shape(data)
            self.data = flat_data
            inferred_strides = None

        if shape is None:
            self.shape = inferred_shape
        else:
            self.shape = tuple(shape)

        self.offset = offset

        if strides is None:
            if inferred_strides is not None and shape is None:
                self.strides = inferred_strides
            else:
                # Calculate C-contiguous strides
                temp_strides = []
                current_stride = 1
                for dim in reversed(self.shape):
                    temp_strides.append(current_stride)
                    current_stride *= dim
                self.strides = tuple(reversed(temp_strides))
        else:
            self.strides = tuple(strides)

        # 🧠 2. Autograd Graph Attributes
        self.requires_grad: bool = requires_grad
        self.grad: Optional['Tensor'] = None
        self.grad_fn: Optional[Any] = None
        self._retains_grad: bool = False

    @property
    def is_leaf(self) -> bool:
        """A tensor is a leaf if it does not have a grad_fn."""
        return self.grad_fn is None

    def retain_grad(self) -> None:
        """Enables gradient retention for non-leaf tensors."""
        self._retains_grad = True

    def zero_grad(self) -> None:
        """Resets the accumulated gradient to None."""
        self.grad = None

    def backward(self, gradient: Optional[Any] = None, retain_graph: bool = False) -> None:
        """Computes the gradient of this tensor w.r.t. graph leaves."""
        from nevula.autograd.engine import backward
        backward(self, grad=gradient, retain_graph=retain_graph)

    # 🔍 3. Coordinate Mapping & Access
    def _flat_index(self, indices: tuple[int, ...]) -> int:
        """Converts multi-dimensional index (i, j, ...) to flat memory index."""
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        flat_idx = self.offset
        for i, idx in enumerate(indices):
            dim_size = self.shape[i]
            if idx < 0:
                idx += dim_size
            if not (0 <= idx < dim_size):
                raise IndexError(f"Index {idx} is out of bounds for dimension {i} with size {dim_size}")
            flat_idx += idx * self.strides[i]
        return flat_idx

    def _indices_generator(self):
        """Generates all multi-dimensional coordinate tuples for self.shape."""
        yield from _generate_indices(self.shape)

    def _broadcast_index(self, broadcast_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Maps an index tuple of a broadcasted shape back to self's shape."""
        ndim = len(self.shape)
        max_ndim = len(broadcast_idx)
        mapped = []
        for i in range(ndim):
            b_idx = broadcast_idx[max_ndim - ndim + i]
            if self.shape[i] == 1:
                mapped.append(0)
            else:
                mapped.append(b_idx)
        return tuple(mapped)

    def __getitem__(self, indices):
        """Allows tensor[i, j] syntax to read values."""
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) == len(self.shape):
            flat_idx = self._flat_index(indices)
            return self.data[flat_idx]
        elif len(indices) < len(self.shape):
            new_offset = self.offset
            for i, idx in enumerate(indices):
                dim_size = self.shape[i]
                if idx < 0:
                    idx += dim_size
                if not (0 <= idx < dim_size):
                    raise IndexError(f"Index {idx} is out of bounds for dimension {i} with size {dim_size}")
                new_offset += idx * self.strides[i]

            new_shape = self.shape[len(indices):]
            new_strides = self.strides[len(indices):]
            return Tensor(self.data, shape=new_shape, strides=new_strides, offset=new_offset, _clone=False)
        else:
            raise IndexError("Too many indices for tensor.")

    def __setitem__(self, indices, value):
        """Allows tensor[i, j] = val syntax to write values."""
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) == len(self.shape):
            flat_idx = self._flat_index(indices)
            self.data[flat_idx] = value
        elif len(indices) < len(self.shape):
            view = self[indices]
            if isinstance(value, Tensor):
                if view.shape != value.shape:
                    raise ValueError(f"Cannot assign tensor of shape {value.shape} to view of shape {view.shape}")
                for idx in view._indices_generator():
                    view[idx] = value[idx]
            elif isinstance(value, (list, tuple)):
                val_tensor = Tensor(value)
                if view.shape != val_tensor.shape:
                    raise ValueError(f"Cannot assign structure of shape {val_tensor.shape} to view of shape {view.shape}")
                for idx in view._indices_generator():
                    view[idx] = val_tensor[idx]
            else:
                for idx in view._indices_generator():
                    view[idx] = value
        else:
            raise IndexError("Too many indices for tensor.")

    def is_contiguous(self) -> bool:
        """Checks if the strides of the tensor are contiguous."""
        if len(self.shape) == 0:
            return True
        expected_strides = []
        current_stride = 1
        for dim in reversed(self.shape):
            expected_strides.append(current_stride)
            current_stride *= dim
        expected_strides = tuple(reversed(expected_strides))
        return self.strides == expected_strides

    def contiguous(self) -> 'Tensor':
        """Returns a contiguous copy of the tensor."""
        if self.is_contiguous() and self.offset == 0 and len(self.data) == _prod(self.shape):
            return self
        new_data = [self[idx] for idx in self._indices_generator()]
        return Tensor(new_data, shape=self.shape)

    def clone(self) -> 'Tensor':
        """Returns a copy of the tensor with distinct data storage."""
        new_data = [self[idx] for idx in self._indices_generator()]
        res = Tensor(new_data, shape=self.shape, requires_grad=self.requires_grad)
        return res

    def detach(self) -> 'Tensor':
        """
        Returns a new Tensor, detached from the current computational graph.
        The returned tensor shares the same underlying data, but has requires_grad=False
        and grad_fn=None.
        """
        return Tensor(
            self.data,
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            _clone=False,
            requires_grad=False,
        )

    # 🔄 4. Zero-Copy View Transformations
    def _reshape_raw(self, new_shape: tuple[int, ...]) -> 'Tensor':
        resolved_shape = list(new_shape)
        if -1 in resolved_shape:
            if resolved_shape.count(-1) > 1:
                raise ValueError("Only one dimension can be -1")
            idx = resolved_shape.index(-1)
            total_elements = _prod(self.shape)
            other_elements = _prod([d for d in resolved_shape if d != -1])
            if other_elements == 0 or total_elements % other_elements != 0:
                raise ValueError(f"Cannot reshape tensor of shape {self.shape} to {new_shape}")
            resolved_shape[idx] = total_elements // other_elements

        resolved_shape = tuple(resolved_shape)

        if _prod(resolved_shape) != _prod(self.shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to {resolved_shape}")

        if self.is_contiguous():
            new_strides = []
            current_stride = 1
            for dim in reversed(resolved_shape):
                new_strides.append(current_stride)
                current_stride *= dim
            new_strides = tuple(reversed(new_strides))
            return Tensor(self.data, shape=resolved_shape, strides=new_strides, offset=self.offset, _clone=False)
        else:
            contiguous_self = self.contiguous()
            return contiguous_self._reshape_raw(resolved_shape)

    def reshape(self, *new_shape) -> 'Tensor':
        """Returns a new Tensor view with a different shape."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            shape = tuple(new_shape[0])
        else:
            shape = tuple(new_shape)

        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Reshape
        if GradMode.is_enabled() and self.requires_grad:
            return Reshape.apply(self, shape)
        return self._reshape_raw(shape)

    def transpose(self, axis1: int, axis2: int) -> 'Tensor':
        """Swaps two axes by simply swapping their shapes and strides."""
        ndim = len(self.shape)
        if not (-ndim <= axis1 < ndim) or not (-ndim <= axis2 < ndim):
            raise IndexError("Dimension out of range")

        if axis1 < 0:
            axis1 += ndim
        if axis2 < 0:
            axis2 += ndim

        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Transpose
        if GradMode.is_enabled() and self.requires_grad:
            return Transpose.apply(self, axis1, axis2)

        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]
        new_strides[axis1], new_strides[axis2] = new_strides[axis2], new_strides[axis1]

        return Tensor(self.data, shape=tuple(new_shape), strides=tuple(new_strides), offset=self.offset, _clone=False)

    @property
    def T(self) -> 'Tensor':
        """Transpose shortcut for 2D matrices."""
        if len(self.shape) < 2:
            return self
        return self.transpose(-2, -1)

    # ➕ 5. Raw Mathematical Operations (internal without autograd)
    def _add_raw(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        for idx in _generate_indices(out_shape):
            s_idx = self._broadcast_index(idx)
            o_idx = other._broadcast_index(idx)
            out_data.append(self[s_idx] + other[o_idx])
        return Tensor(out_data, shape=out_shape)

    def _sub_raw(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        for idx in _generate_indices(out_shape):
            s_idx = self._broadcast_index(idx)
            o_idx = other._broadcast_index(idx)
            out_data.append(self[s_idx] - other[o_idx])
        return Tensor(out_data, shape=out_shape)

    def _mul_raw(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        for idx in _generate_indices(out_shape):
            s_idx = self._broadcast_index(idx)
            o_idx = other._broadcast_index(idx)
            out_data.append(self[s_idx] * other[o_idx])
        return Tensor(out_data, shape=out_shape)

    def _div_raw(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        for idx in _generate_indices(out_shape):
            s_idx = self._broadcast_index(idx)
            o_idx = other._broadcast_index(idx)
            out_data.append(self[s_idx] / other[o_idx])
        return Tensor(out_data, shape=out_shape)

    def _pow_raw(self, p: Any) -> 'Tensor':
        p_val = p.to_list() if isinstance(p, Tensor) else p
        out_data = [self[idx] ** p_val for idx in self._indices_generator()]
        return Tensor(out_data, shape=self.shape)

    def _neg_raw(self) -> 'Tensor':
        out_data = [-self[idx] for idx in self._indices_generator()]
        return Tensor(out_data, shape=self.shape)

    def _matmul_raw(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Standard 2D matrix multiplication
        if len(self.shape) == 2 and len(other.shape) == 2:
            m, k1 = self.shape
            k2, n = other.shape
            if k1 != k2:
                raise ValueError(f"Cannot multiply matrices with shapes {self.shape} and {other.shape}")

            out_data = []
            for i in range(m):
                for j in range(n):
                    cell = 0.0
                    for k in range(k1):
                        cell += self[i, k] * other[k, j]
                    out_data.append(cell)
            return Tensor(out_data, shape=(m, n))

        # 1D dot product
        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Cannot compute dot product with shapes {self.shape} and {other.shape}")
            total = sum(self[i] * other[i] for i in range(self.shape[0]))
            return Tensor(total)

        # Matrix-vector or Vector-matrix or Batched
        if len(self.shape) == 2 and len(other.shape) == 1:
            m, k1 = self.shape
            if k1 != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            out_data = [sum(self[i, k] * other[k] for k in range(k1)) for i in range(m)]
            return Tensor(out_data, shape=(m,))

        if len(self.shape) == 1 and len(other.shape) == 2:
            k1, n = other.shape
            if self.shape[0] != k1:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            out_data = [sum(self[k] * other[k, j] for k in range(k1)) for j in range(n)]
            return Tensor(out_data, shape=(n,))

        raise NotImplementedError(f"MatMul between shapes {self.shape} and {other.shape} not yet supported.")

    def _sum_raw(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        if axis is None:
            total = sum(self[idx] for idx in self._indices_generator())
            if keepdims:
                return Tensor([total], shape=(1,) * len(self.shape))
            return Tensor(total)

        ndim = len(self.shape)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes_set = {ax if ax >= 0 else ax + ndim for ax in axes}

        if keepdims:
            out_shape = tuple(1 if i in axes_set else self.shape[i] for i in range(ndim))
        else:
            out_shape = tuple(self.shape[i] for i in range(ndim) if i not in axes_set)
            if len(out_shape) == 0:
                out_shape = ()

        acc: dict[tuple[int, ...], float] = {}
        for idx in self._indices_generator():
            if keepdims:
                out_idx = tuple(0 if i in axes_set else idx[i] for i in range(ndim))
            else:
                out_idx = tuple(idx[i] for i in range(ndim) if i not in axes_set)
            acc[out_idx] = acc.get(out_idx, 0.0) + self[idx]

        out_data = [acc[out_idx] for out_idx in _generate_indices(out_shape)]
        return Tensor(out_data, shape=out_shape)

    def _mean_raw(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        s = self._sum_raw(axis=axis, keepdims=keepdims)
        num_elements = _prod(self.shape) // max(1, _prod(s.shape))
        return s._div_raw(Tensor(float(num_elements)))

    # 🔗 6. Public Operator Overloads with Autograd Hooking
    def __add__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Add
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            return Add.apply(self, other)
        return self._add_raw(other)

    def __radd__(self, other: Any) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Sub
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            return Sub.apply(self, other)
        return self._sub_raw(other)

    def __rsub__(self, other: Any) -> 'Tensor':
        return Tensor(other).__sub__(self)

    def __mul__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Mul
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            return Mul.apply(self, other)
        return self._mul_raw(other)

    def __rmul__(self, other: Any) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Div
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            return Div.apply(self, other)
        return self._div_raw(other)

    def __rtruediv__(self, other: Any) -> 'Tensor':
        return Tensor(other).__truediv__(self)

    def __pow__(self, p: Any) -> 'Tensor':
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Pow
        if GradMode.is_enabled() and self.requires_grad:
            return Pow.apply(self, p)
        return self._pow_raw(p)

    def __neg__(self) -> 'Tensor':
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Neg
        if GradMode.is_enabled() and self.requires_grad:
            return Neg.apply(self)
        return self._neg_raw()

    def __matmul__(self, other: Any) -> 'Tensor':
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> 'Tensor':
        return Tensor(other).matmul(self)

    def matmul(self, other: Any) -> 'Tensor':
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import MatMul
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            return MatMul.apply(self, other)
        return self._matmul_raw(other)

    def sum(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        """Computes the sum along specified axis or all dimensions."""
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Sum
        if GradMode.is_enabled() and self.requires_grad:
            return Sum.apply(self, axis, keepdims)
        return self._sum_raw(axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        """Computes the arithmetic mean along specified axis or all dimensions."""
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Mean
        if GradMode.is_enabled() and self.requires_grad:
            return Mean.apply(self, axis, keepdims)
        return self._mean_raw(axis=axis, keepdims=keepdims)

    def relu(self) -> 'Tensor':
        """Applies rectified linear unit elementwise."""
        from nevula.autograd.functions import ReLU
        return ReLU.apply(self)

    def sigmoid(self) -> 'Tensor':
        """Applies sigmoid elementwise."""
        from nevula.autograd.functions import Sigmoid
        return Sigmoid.apply(self)

    def tanh(self) -> 'Tensor':
        """Applies hyperbolic tangent elementwise."""
        from nevula.autograd.functions import Tanh
        return Tanh.apply(self)

    # 🏭 7. Factory Methods
    @classmethod
    def zeros(cls, shape: tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Creates a tensor of all zeros with the given shape."""
        num = _prod(shape)
        return cls([0.0] * num, shape=tuple(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Creates a tensor of all ones with the given shape."""
        num = _prod(shape)
        return cls([1.0] * num, shape=tuple(shape), requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, other: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """Creates a tensor of zeros matching the shape of another tensor."""
        return cls.zeros(other.shape, requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, other: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """Creates a tensor of ones matching the shape of another tensor."""
        return cls.ones(other.shape, requires_grad=requires_grad)

    def item(self) -> Any:
        """Returns the value of this tensor as a standard Python number."""
        if _prod(self.shape) != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self.data[self.offset]

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def to_list(self):
        """Converts the tensor to a nested Python list."""
        if len(self.shape) == 0:
            return self.data[self.offset]

        def convert(idx_prefix):
            dim = len(idx_prefix)
            if dim == len(self.shape) - 1:
                return [self[idx_prefix + (i,)] for i in range(self.shape[dim])]
            return [convert(idx_prefix + (i,)) for i in range(self.shape[dim])]

        return convert(())

    def __repr__(self) -> str:
        """Clean string representation for printing."""
        extra = []
        if self.grad_fn is not None:
            extra.append(f"grad_fn={self.grad_fn}")
        elif self.requires_grad:
            extra.append("requires_grad=True")
        extra_str = f", {', '.join(extra)}" if extra else ""
        return f"Tensor({self.to_list()}, shape={self.shape}{extra_str})"