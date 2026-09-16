from typing import Any, Optional, Sequence, Union
import numpy as np
from nevula.backend.device import Device
from nevula.backend.registry import get_backend


def _prod(shape: tuple[int, ...]) -> int:
    """Calculates the product of a shape tuple."""
    res = 1
    for x in shape:
        res *= x
    return res


def _record_graph(op: str, inputs: Sequence[Any], output: Any, **kwargs) -> None:
    try:
        from nevula.graph.capture import GraphCapture
        if GraphCapture.is_active():
            GraphCapture.record_op(op, inputs, output, **kwargs)
    except (ImportError, Exception):
        pass



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
        device: Union[str, Device] = "cpu",
    ):
        # 📱 Device resolution
        if isinstance(device, Device):
            self.device = device
        else:
            self.device = Device(device)

        backend = get_backend(self.device.type)

        # 📦 1. Storage & Layout Metadata
        if isinstance(data, Tensor):
            self.data = data.data
            inferred_shape = data.shape
            inferred_strides = data.strides
            offset = data.offset
            if device == "cpu" and data.device != "cpu":
                self.device = data.device
        elif not _clone:
            self.data = data
            inferred_shape = tuple(shape) if shape is not None else getattr(data, "shape", (len(data),) if hasattr(data, "__len__") else ())
            inferred_strides = strides
        elif isinstance(data, np.ndarray):
            inferred_shape = data.shape
            self.data = backend.create(data.flatten())
            inferred_strides = None
        else:
            flat_data, inferred_shape = _flatten_and_infer_shape(data)
            self.data = backend.create(flat_data)
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
        self.name: Optional[str] = None


    def _check_same_device(self, other: Any) -> None:
        """Verifies that other tensor resides on the same device."""
        if isinstance(other, Tensor):
            if self.device != other.device:
                raise RuntimeError(
                    f"Expected all tensors to be on the same device, but found {self.device.type} and {other.device.type}."
                )

    def to_array(self) -> Any:
        """Returns a backend array view reflecting shape, strides, and offset."""
        if self.device.type == "cpu":
            itemsize = self.data.itemsize
            byte_offset = self.offset * itemsize
            byte_strides = tuple(s * itemsize for s in self.strides)
            return np.ndarray(self.shape, dtype=self.data.dtype, buffer=self.data, offset=byte_offset, strides=byte_strides)
        elif self.device.type == "cuda":
            try:
                import cupy as cp
                itemsize = self.data.itemsize
                byte_offset = self.offset * itemsize
                byte_strides = tuple(s * itemsize for s in self.strides)
                memptr = self.data.data + byte_offset
                return cp.ndarray(self.shape, dtype=self.data.dtype, memptr=memptr, strides=byte_strides)
            except Exception:
                itemsize = self.data.itemsize
                byte_offset = self.offset * itemsize
                byte_strides = tuple(s * itemsize for s in self.strides)
                return np.ndarray(self.shape, dtype=self.data.dtype, buffer=self.data, offset=byte_offset, strides=byte_strides)
        else:
            raise NotImplementedError(f"to_array not implemented for device {self.device}")

    def to(self, device: Union[str, Device]) -> 'Tensor':
        """
        Moves tensor to specified compute device ('cpu', 'cuda').
        Returns self if already on target device, or new Tensor on target device.
        """
        target_device = Device(device)
        if self.device == target_device:
            return self

        from nevula.nn.parameter import Parameter
        if isinstance(self, Parameter):
            src_backend = get_backend(self.device.type)
            dst_backend = get_backend(target_device.type)
            src_arr = self.to_array()
            np_arr = src_backend.to_numpy(src_arr)
            dst_data = dst_backend.from_numpy(np_arr)
            self.data = dst_data.flatten()
            self.device = target_device
            if self.grad is not None:
                self.grad = self.grad.to(target_device)
            return self

        src_backend = get_backend(self.device.type)
        dst_backend = get_backend(target_device.type)
        src_arr = self.to_array()
        np_arr = src_backend.to_numpy(src_arr)
        dst_data = dst_backend.from_numpy(np_arr)

        return Tensor(
            dst_data,
            requires_grad=self.requires_grad,
            device=target_device,
        )


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
            val = self.data[flat_idx]
            return val.item() if hasattr(val, "item") else val
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
            return Tensor(self.data, shape=new_shape, strides=new_strides, offset=new_offset, _clone=False, device=self.device)
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
                val_tensor = Tensor(value, device=self.device)
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
        return Tensor(new_data, shape=self.shape, device=self.device)

    def clone(self) -> 'Tensor':
        """Returns a copy of the tensor with distinct data storage."""
        new_data = [self[idx] for idx in self._indices_generator()]
        res = Tensor(new_data, shape=self.shape, requires_grad=self.requires_grad, device=self.device)
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
            device=self.device,
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
            return Tensor(self.data, shape=resolved_shape, strides=new_strides, offset=self.offset, _clone=False, device=self.device)
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
            res = Reshape.apply(self, shape)
        else:
            res = self._reshape_raw(shape)
        _record_graph("reshape", [self], res, shape=shape)
        return res

    def _transpose_raw(self, axis1: int, axis2: int) -> 'Tensor':
        ndim = len(self.shape)
        if axis1 < 0:
            axis1 += ndim
        if axis2 < 0:
            axis2 += ndim
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]
        new_strides[axis1], new_strides[axis2] = new_strides[axis2], new_strides[axis1]
        return Tensor(self.data, shape=tuple(new_shape), strides=tuple(new_strides), offset=self.offset, _clone=False, device=self.device)

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
            res = Transpose.apply(self, axis1, axis2)
        else:
            res = self._transpose_raw(axis1, axis2)

        _record_graph("transpose", [self], res, axis1=axis1, axis2=axis2)
        return res



    @property
    def T(self) -> 'Tensor':
        """Transpose shortcut for 2D matrices."""
        if len(self.shape) < 2:
            return self
        return self.transpose(-2, -1)

    # ➕ 5. Raw Mathematical Operations (internal without autograd)
    # ➕ 5. Raw Mathematical Operations (delegated to backend)
    def _add_raw(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        backend = get_backend(self.device.type)
        out = backend.add(self.to_array(), other.to_array())
        return Tensor(out, device=self.device)

    def _sub_raw(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        backend = get_backend(self.device.type)
        out = backend.sub(self.to_array(), other.to_array())
        return Tensor(out, device=self.device)

    def _mul_raw(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        backend = get_backend(self.device.type)
        out = backend.mul(self.to_array(), other.to_array())
        return Tensor(out, device=self.device)

    def _div_raw(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        backend = get_backend(self.device.type)
        out = backend.div(self.to_array(), other.to_array())
        return Tensor(out, device=self.device)

    def _pow_raw(self, p: Any) -> 'Tensor':
        backend = get_backend(self.device.type)
        if isinstance(p, Tensor):
            self._check_same_device(p)
            p_val = p.to_array()
        else:
            p_val = p
        out = backend.pow(self.to_array(), p_val)
        return Tensor(out, device=self.device)

    def _neg_raw(self) -> 'Tensor':
        backend = get_backend(self.device.type)
        out = backend.neg(self.to_array())
        return Tensor(out, device=self.device)

    def _matmul_raw(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        backend = get_backend(self.device.type)
        out = backend.matmul(self.to_array(), other.to_array())
        return Tensor(out, device=self.device)

    def _sum_raw(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        backend = get_backend(self.device.type)
        out = backend.sum(self.to_array(), axis=axis, keepdims=keepdims)
        return Tensor(out, device=self.device)

    def _mean_raw(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        backend = get_backend(self.device.type)
        out = backend.mean(self.to_array(), axis=axis, keepdims=keepdims)
        return Tensor(out, device=self.device)

    # 🔗 6. Public Operator Overloads with Autograd Hooking
    def __add__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Add
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            res = Add.apply(self, other)
        else:
            res = self._add_raw(other)
        _record_graph("add", [self, other], res)
        return res

    def __radd__(self, other: Any) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Sub
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            res = Sub.apply(self, other)
        else:
            res = self._sub_raw(other)
        _record_graph("sub", [self, other], res)
        return res

    def __rsub__(self, other: Any) -> 'Tensor':
        return Tensor(other, device=self.device).__sub__(self)

    def __mul__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Mul
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            res = Mul.apply(self, other)
        else:
            res = self._mul_raw(other)
        _record_graph("mul", [self, other], res)
        return res

    def __rmul__(self, other: Any) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Div
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            res = Div.apply(self, other)
        else:
            res = self._div_raw(other)
        _record_graph("div", [self, other], res)
        return res

    def __rtruediv__(self, other: Any) -> 'Tensor':
        return Tensor(other, device=self.device).__truediv__(self)

    def __pow__(self, p: Any) -> 'Tensor':
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Pow
        if GradMode.is_enabled() and self.requires_grad:
            res = Pow.apply(self, p)
        else:
            res = self._pow_raw(p)
        _record_graph("pow", [self, p], res)
        return res

    def __neg__(self) -> 'Tensor':
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Neg
        if GradMode.is_enabled() and self.requires_grad:
            res = Neg.apply(self)
        else:
            res = self._neg_raw()
        _record_graph("neg", [self], res)
        return res

    def __matmul__(self, other: Any) -> 'Tensor':
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> 'Tensor':
        return Tensor(other, device=self.device).matmul(self)

    def matmul(self, other: Any) -> 'Tensor':
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        self._check_same_device(other)
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import MatMul
        if GradMode.is_enabled() and (self.requires_grad or other.requires_grad):
            res = MatMul.apply(self, other)
        else:
            res = self._matmul_raw(other)
        _record_graph("matmul", [self, other], res)
        return res

    def sum(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        """Computes the sum along specified axis or all dimensions."""
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Sum
        if GradMode.is_enabled() and self.requires_grad:
            res = Sum.apply(self, axis, keepdims)
        else:
            res = self._sum_raw(axis=axis, keepdims=keepdims)
        _record_graph("sum", [self], res, axis=axis, keepdims=keepdims)
        return res

    def mean(self, axis: Optional[Any] = None, keepdims: bool = False) -> 'Tensor':
        """Computes the arithmetic mean along specified axis or all dimensions."""
        from nevula.autograd.engine import GradMode
        from nevula.autograd.functions import Mean
        if GradMode.is_enabled() and self.requires_grad:
            res = Mean.apply(self, axis, keepdims)
        else:
            res = self._mean_raw(axis=axis, keepdims=keepdims)
        _record_graph("mean", [self], res, axis=axis, keepdims=keepdims)
        return res

    def relu(self) -> 'Tensor':
        """Applies rectified linear unit elementwise."""
        from nevula.autograd.functions import ReLU
        res = ReLU.apply(self)
        _record_graph("relu", [self], res)
        return res

    def sigmoid(self) -> 'Tensor':
        """Applies sigmoid elementwise."""
        from nevula.autograd.functions import Sigmoid
        res = Sigmoid.apply(self)
        _record_graph("sigmoid", [self], res)
        return res

    def tanh(self) -> 'Tensor':
        """Applies hyperbolic tangent elementwise."""
        from nevula.autograd.functions import Tanh
        res = Tanh.apply(self)
        _record_graph("tanh", [self], res)
        return res

    def abs(self) -> 'Tensor':
        """Applies absolute value elementwise."""
        from nevula.autograd.functions import Abs
        res = Abs.apply(self)
        _record_graph("abs", [self], res)
        return res

    def __abs__(self) -> 'Tensor':
        return self.abs()


    # 🏭 7. Factory Methods
    @classmethod
    def zeros(cls, shape: tuple[int, ...], requires_grad: bool = False, device: Union[str, Device] = "cpu") -> 'Tensor':
        """Creates a tensor of all zeros with the given shape on the specified device."""
        num = _prod(shape)
        return cls([0.0] * num, shape=tuple(shape), requires_grad=requires_grad, device=device)

    @classmethod
    def ones(cls, shape: tuple[int, ...], requires_grad: bool = False, device: Union[str, Device] = "cpu") -> 'Tensor':
        """Creates a tensor of all ones with the given shape on the specified device."""
        num = _prod(shape)
        return cls([1.0] * num, shape=tuple(shape), requires_grad=requires_grad, device=device)

    @classmethod
    def zeros_like(cls, other: 'Tensor', requires_grad: bool = False, device: Optional[Union[str, Device]] = None) -> 'Tensor':
        """Creates a tensor of zeros matching the shape and device of another tensor."""
        dev = other.device if device is None else device
        return cls.zeros(other.shape, requires_grad=requires_grad, device=dev)

    @classmethod
    def ones_like(cls, other: 'Tensor', requires_grad: bool = False, device: Optional[Union[str, Device]] = None) -> 'Tensor':
        """Creates a tensor of ones matching the shape and device of another tensor."""
        dev = other.device if device is None else device
        return cls.ones(other.shape, requires_grad=requires_grad, device=dev)

    def item(self) -> Any:
        """Returns the value of this tensor as a standard Python number."""
        if _prod(self.shape) != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        val = self.data[self.offset]
        return val.item() if hasattr(val, "item") else val

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def to_list(self):
        """Converts the tensor to a nested Python list."""
        if len(self.shape) == 0:
            val = self.data[self.offset]
            return val.item() if hasattr(val, "item") else val

        def convert(idx_prefix):
            dim = len(idx_prefix)
            if dim == len(self.shape) - 1:
                return [
                    self[idx_prefix + (i,)].item()
                    if hasattr(self[idx_prefix + (i,)], "item")
                    else self[idx_prefix + (i,)]
                    for i in range(self.shape[dim])
                ]
            return [convert(idx_prefix + (i,)) for i in range(self.shape[dim])]

        return convert(())

    def __repr__(self) -> str:
        """Clean string representation for printing."""
        extra = []
        if self.device.type != "cpu":
            extra.append(f"device='{self.device.type}'")
        if self.grad_fn is not None:
            extra.append(f"grad_fn={self.grad_fn}")
        elif self.requires_grad:
            extra.append("requires_grad=True")
        extra_str = f", {', '.join(extra)}" if extra else ""
        return f"Tensor({self.to_list()}, shape={self.shape}{extra_str})"