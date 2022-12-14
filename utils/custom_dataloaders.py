import torch
from torchvision import datasets
from torchvision import transforms 


# Load MNIST/CIFAR10
def load_MNIST(batch_size):
    train_dataset = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = transforms.ToTensor(), 
    )

    test_dataset = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = transforms.ToTensor(), 
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preprocessing_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,)) ,
        ])
    return train_dataloader, test_dataloader, preprocessing_transform