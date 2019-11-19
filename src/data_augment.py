from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

def data_import(path):
    transform_data = transforms.Compose([
        transform.resize((224, 224))
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = datasets.ImageFolder(root=path, transform=transform_train)
    return data_train, data_test