import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

a = torch.tensor([2, 3, 4])
b = torch.tensor([1, 5, 2])

c = a + b  
print("Hasil penjumlahan tensor:", c.numpy())  

data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  
labels = torch.tensor([[2.0], [4.0], [6.0], [8.0]]) 

model = nn.Linear(1, 1)  

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) 
epochs = 500
for epoch in range(epochs):
    predictions = model(data)
    loss = criterion(predictions, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    test_input = torch.tensor([[5.0]]) 
    result = model(test_input)
    print("Hasil prediksi untuk input 5:", result.item()) 
