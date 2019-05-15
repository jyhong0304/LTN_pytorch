import logictensornetworks as ltn
from logictensornetworks import And,Not,Forall,Implies
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# loading data

data = np.random.uniform([-1,-1],[1,1],(500,2),).astype(np.float32)

# defining the language

x = ltn.variable("x", data)
y = ltn.variable("y", data)

a = ltn.constant("a",[0.5,0.5])
b = ltn.constant("b",[-0.5,-0.5])

A = ltn.Predicate("A",2)
B = ltn.Predicate("B",2)

params = list(A.parameters()) + list(B.parameters())
optimizer = optim.SGD(params, lr=0.01)
criterion = nn.MSELoss()
optimizer.zero_grad()
output = And(A(a), B(b), Not(A(b)), Forall(x, Implies(A(x), B(x))))
loss = criterion(output, torch.ones_like(output))
while loss == 1.0:
    A.reset_parameters()
    B.reset_parameters()
    params = list(A.parameters()) + list(B.parameters())
    optimizer = optim.RMSprop(params)
    optimizer.zero_grad()
    output = And(A(a), B(b), Not(A(b)), Forall(x, Implies(A(x), B(x))))
    loss = criterion(output, torch.ones_like(output))
for i in range(1000):
    optimizer.zero_grad()
    output = And(A(a), B(b), Not(A(b)), Forall(x, Implies(A(x), B(x))))
    loss = criterion(output, torch.ones_like(output))
    if i % 100 == 0:
        print(loss)
    loss.backward()
    optimizer.step()

# show results:

Aa = A(a)
print("A(a): ", Aa)
Bb = B(b)
print("B(b): ", Bb)
Ba = B(a)
print("B(a): ", Ba)
Ab = A(b)
print("A(b): ", Ab)

