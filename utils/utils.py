import torch
import torch.nn as nn
import torch.optim as optim
import  torch.nn.functional as F
from utils.plots import loss_curve

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, num_epochs, learning_rate):
  model = model.to(DEVICE)
  critirion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  loss_result = []
  for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)
      label_predicted = model(inputs)
      loss = critirion(label_predicted, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        loss_result.append(loss.item())
  loss_curve(loss_result)
  return model



def test(model, test_loader):
  model.eval()
  n_correct = 0
  n_samples = 0
  for inputs, labels in test_loader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    label_predicted = model(inputs)
    _, predicted = torch.max(label_predicted, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
  print(f'Accuracy = {100*n_correct/n_samples:.2f}')
