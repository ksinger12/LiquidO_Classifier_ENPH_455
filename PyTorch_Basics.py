import torch
import numpy as np

# Creating tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Creating tensor from NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Creating tensor from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes -> shape, datatype, device where stored
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Move tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Indexing and slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# Joining Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Aggregating all values of a tensor into one value and converting to numerical value
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In place operations -> use is discouraged since loss in history
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# In this case, a change in the tensor will also be reflected in the numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
