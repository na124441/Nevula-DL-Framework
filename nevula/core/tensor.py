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


class Tensor:
    def __init__(self, data, shape=None, strides=None, offset=0, _clone=True):
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

    # 🔍 2. Coordinate Mapping & Access
    def _flat_index(self, indices: tuple[int, ...]) -> int:
        """Converts multi-dimensional index (i, j, ...) to flat memory index."""
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        
        flat_idx = self.offset
        for i, idx in enumerate(indices):
            dim_size = self.shape[i]
            # Handle negative indices
            if idx < 0:
                idx += dim_size
            if not (0 <= idx < dim_size):
                raise IndexError(f"Index {idx} is out of bounds for dimension {i} with size {dim_size}")
            flat_idx += idx * self.strides[i]
        return flat_idx

    def _indices_generator(self):
        """Generates all multi-dimensional coordinate tuples for self.shape."""
        def gen(dim):
            if dim == len(self.shape):
                yield ()
                return
            for i in range(self.shape[dim]):
                for rest in gen(dim + 1):
                    yield (i,) + rest
        yield from gen(0)

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
            # Return a sub-tensor view
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
            # Assigning to a sub-tensor view
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
                # Assume scalar broadcasting
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

    # 🔄 3. Zero-Copy View Transformations
    def reshape(self, new_shape: tuple[int, ...]) -> 'Tensor':
        """Returns a new Tensor view with a different shape."""
        # Resolve -1 in new_shape
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
            # Calculate contiguous strides for resolved_shape
            new_strides = []
            current_stride = 1
            for dim in reversed(resolved_shape):
                new_strides.append(current_stride)
                current_stride *= dim
            new_strides = tuple(reversed(new_strides))
            return Tensor(self.data, shape=resolved_shape, strides=new_strides, offset=self.offset, _clone=False)
        else:
            # Fallback to copy for non-contiguous tensors
            contiguous_self = self.contiguous()
            return contiguous_self.reshape(resolved_shape)

    def transpose(self, axis1: int, axis2: int) -> 'Tensor':
        """Swaps two axes by simply swapping their shapes and strides."""
        ndim = len(self.shape)
        if not (-ndim <= axis1 < ndim) or not (-ndim <= axis2 < ndim):
            raise IndexError("Dimension out of range")
        
        if axis1 < 0:
            axis1 += ndim
        if axis2 < 0:
            axis2 += ndim
            
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        
        new_shape[axis1], new_shape[axis2] = new_shape[axis2], new_shape[axis1]
        new_strides[axis1], new_strides[axis2] = new_strides[axis2], new_strides[axis1]
        
        return Tensor(self.data, shape=tuple(new_shape), strides=tuple(new_strides), offset=self.offset, _clone=False)

    # ➕ 4. Arithmetic Operations
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        
        # Generator for out_shape coordinates
        def gen(dim):
            if dim == len(out_shape):
                yield ()
                return
            for i in range(out_shape[dim]):
                for rest in gen(dim + 1):
                    yield (i,) + rest
                    
        for idx in gen(0):
            self_idx = self._broadcast_index(idx)
            other_idx = other._broadcast_index(idx)
            out_data.append(self[self_idx] + other[other_idx])
            
        return Tensor(out_data, shape=out_shape)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        out_shape = broadcast_shapes(self.shape, other.shape)
        out_data = []
        
        def gen(dim):
            if dim == len(out_shape):
                yield ()
                return
            for i in range(out_shape[dim]):
                for rest in gen(dim + 1):
                    yield (i,) + rest
                    
        for idx in gen(0):
            self_idx = self._broadcast_index(idx)
            other_idx = other._broadcast_index(idx)
            out_data.append(self[self_idx] * other[other_idx])
            
        return Tensor(out_data, shape=out_shape)

    def __rmul__(self, other):
        return self.__mul__(other)

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

    def __repr__(self):
        """Clean string representation for printing."""
        return f"Tensor({self.to_list()}, shape={self.shape}, strides={self.strides})"