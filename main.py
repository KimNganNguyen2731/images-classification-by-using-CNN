import yaml
from data.CIFAR10 import dataloader
from models.CNN import ConvNet
from utils.utils import train, test

def main():
  with open("config.yml","r") as f:
    config = yaml.safe_load(f)
  model_config = config["model_config")
  train_config = config["train_config"]
  model = ConvNet(
    in_channels = model_config["in_channels"],
    out_channels = model_config["out_channels"],
    img_size = model_config["image_size"],
    output_classes = model_config["output_classes"],
    kernel_size = model_config["kernel_size"],
    padding = model_config["padding"],
    stride = model_config["stride']
                 )
    
  train_loader, test_loader = dataloader()
  trained_model = train(model = model, train_loader = train_loader, num_epochs = train_config["num_epochs"], learning_rate = train_config["lr"])
  test(model = trained_model, test_loader = test_loader)


if __name__ == "__main__":
  main()
