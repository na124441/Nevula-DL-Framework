from nevula.core.tensor import Tensor

# 1. Creating Tensors (supports scalars, 1D, 2D, and higher-dimensional nested lists)
x = Tensor([[1, 2, 3], [4, 5, 6]])
print(f"Tensor:\n{x}")
print(f"Shape: {x.shape}, Strides: {x.strides}\n")

# 2. Reading and Writing Elements
print(f"Element at [1, 2]: {x[1, 2]}")
print(f"Element with negative indices [ -1, -1 ]: {x[-1, -1]}")
x[0, 1] = 99
print(f"After modifying [0, 1]:\n{x}\n")

# 3. Getting Sub-Tensor Views (Partial Indexing)
row = x[1]  # Returns a 1D tensor view of the second row
print(f"Sub-tensor view x[1]: {row}")
print(f"Sub-tensor Offset: {row.offset}\n")

# 4. Zero-Copy View Transformations
# Transposing swaps axes and strides without copying data
x_t = x.transpose(0, 1)
print(f"Transposed:\n{x_t}")
print(f"Transposed Strides: {x_t.strides}")

# Because it is a view, modifying x_t also updates x
x_t[1, 0] = 500
print(f"After modifying transpose view, original x is:\n{x}\n")

# Reshaping is zero-copy when contiguous (supports -1 inference)
x_reshaped = x.reshape((3, 2))
print(f"Reshaped (3, 2):\n{x_reshaped}\n")

# 5. Broadcasting Arithmetic Operations
a = Tensor([[1, 2], [3, 4]])
b = Tensor([10, 20])  # Broadcasts across rows
c = a + b
print(f"Addition with Broadcasting:\n{c}")

d = a * 5  # Scalar multiplication
print(f"Scalar Multiplication:\n{d}")
