import kagglehub
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)

import argparse

def lisa_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    path = kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")
    train_dir = f"{path}/cropped_lisa_1/train_1"
    val_dir = f"{path}/cropped_lisa_1/val_1"

    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, 7


def bstl_loaders(train_batch_size=256, test_batch_size=64, normalize=False):
    path = kagglehub.dataset_download("andrevinic/bstl-dataset")
    train_dir = f"{path}/train"
    test_dir = f"{path}/test"
    
    transform_list = [transforms.Resize((64, 32)), transforms.ToTensor()]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform_list)

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader, 4

def get_dataset(dataset_name,
                train_batch_size=256,
                test_batch_size=64,
                normalize=False):

    if dataset_name == "lisa":
        return lisa_loaders(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            normalize=normalize
        )

    elif dataset_name == "bstl":
        return bstl_loaders(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            normalize=normalize
        )

    else:
        raise ValueError(f"Dataset não suportado: {dataset_name}")

def get_model(model_name, num_classes):
    if model_name == "res":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "eff":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='bstl', choices=['bstl', 'lisa'], help='Dataset a ser utilizado')
    parser.add_argument('-m', '--model', type=str, default='res', choices=['res', 'eff'], help='Modelo a ser utilizado')
    parser.add_argument('-e', '--epochs', type=int, default=40, help='Número de épocas do treinamento')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size de treinamento')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('-ts', '--test_batch_size', type=int, default=256, help='Batch size de teste')
    parser.add_argument('-norm', '--normalize', action='store_true', help='Aplica normalização ImageNet')
    parser.add_argument('-ignore_aa', '--ignore_autoattack', action='store_true')
    parser.add_argument('-tl', '--total_loops', type=int, default=1)

    return parser.parse_args()