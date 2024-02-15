randint = torch.randint(-100,100, (2,3))

torch_tensor = torch.Tensor(randint)

zeros = torch.zeros(2,3)

ones = torch.ones(2,3)

input_ = torch.empty(2,3)

range_ = torch.arange(1, 12, 2)

linspace = torch.linspace(1,100, steps=5)

logspace = torch.logspace(1,10, steps=5)

eye = torch.eye(5)

a = torch.empty(2,3, dtype=torch.int64)
empty_like = torch.empty_like(a)

probabilities = torch.tensor([0.2,0.2,0.6])
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)    # probability of getting 0 => 20%, 1 => 20%, 2 => 80%
# ===>> tensor([1, 0, 2, 0, 2, 2, 2, 2, 0, 2])

tensor = torch.tensor([1,2,3,4])
out = torch.cat((tensor, torch.tensor([5])), dim=0)

out = torch.tril(torch.ones(5,5))

out = torch.triu(torch.ones(5,5))



# make a matrix of ones and them make it upper triangle and then fill the zeros with '-inf' and later apply this mask on the original matrix
out = torch.zeros(5,5).masked_fill(torch.tril(torch.ones(5,5)) != 0, float('-inf'))
print(torch.tril(torch.ones(5,5)) != 0)
print(out)
#tensor([[ True, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True, False, False],
#         [ True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True]])
# tensor([[-inf, 0., 0., 0., 0.],
#         [-inf, -inf, 0., 0., 0.],
#         [-inf, -inf, -inf, 0., 0.],
#         [-inf, -inf, -inf, -inf, 0.],
#         [-inf, -inf, -inf, -inf, -inf]])

torch.exp(out)
# Output==> tensor([[1., 0., 0., 0., 0.],
#        [1., 1., 0., 0., 0.],
#        [1., 1., 1., 0., 0.],
#        [1., 1., 1., 1., 0.],
#        [1., 1., 1., 1., 1.]])



# Suppose you have a tensor with the shape (A, B, C). If you perform a transpose operation with (0, 2), 
# the dimensions will be reordered as (C, B, A). So, the original dimension 0 becomes the new dimension 2, 
# and the original dimension 2 becomes the new dimension 0. The dimensions in between stay the same.
input_ = torch.zeros(2,3,4)
out = input_.transpose(0,2)
out.shape
# ==>> torch.Size([4, 3, 2])


tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([4,5,6])
tensor3 = torch.tensor([7,8,9])
stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
# tensor([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])


import torch.nn.functional as F
tensor1 = torch.tensor([1.0,2.0,3.0])
softmax_output = F.softmax(tensor1, dim=0)
print(softmax_output)
# Output==> tensor([0.0900, 0.2447, 0.6652])

sum = np.exp(1) + np.exp(2) + np.exp(3)
print(np.exp(1)/sum , np.exp(2)/sum, np.exp(3)/sum)
# Output==> 0.0900 0.2447 0.6652


embedding = nn.Embedding(num_embeddings=80, embedding_dim=6) # num_embeddings or the vocabSize
embedded_output = embedding(torch.LongTensor([1,5,3,2]))  # the entry element are the input indicies
print(embedded_output.shape)
# torch.Size([4, 6])


