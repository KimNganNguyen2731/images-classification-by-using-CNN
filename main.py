from data.CIFAR10 import dataloader
from models.CNN import ConvNet
from utils.utils import train, test

def main():
  train_loader, test_loader = dataloader()
  trained_model = train()
  test()


if __name__ == "__main__":
  main()
