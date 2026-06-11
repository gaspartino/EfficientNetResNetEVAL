import os
import torch
import torch.nn as nn
import torch.optim as optim
from train import train_model
from utils import (
    get_args, 
    get_dataset, 
    get_model
)
from attack import (
    accuracy_clean,
    accuracy_FGSM,
    accuracy_PGD,
    accuracy_MIM,
    accuracy_AutoAttack
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_args()

train_loader, test_loader, num_classes = get_dataset(args.dataset, args.batch_size, args.test_batch_size, args.normalize)

for loop_idx in range(args.total_loops):
        
    if args.total_loops > 1:
        print(f"\n========== Loop {loop_idx + 1}/{args.total_loops} ==========\n")

    model = get_model(args.model, num_classes).to(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(args.epochs, model, train_loader, criterion, optimizer, device)

    save_dir = f"saved_models/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{args.model}_{loop_idx+1}.pth"

    torch.save(model.state_dict(), save_path)

    print(f"Modelo salvo em {save_path}")

all_eps = [0.01, 8/255, 0.04, 0.055, 0.07, 0.085, 0.1, 0.115, 0.13, 0.15, 0.175, 0.2]
for loop_idx in range(args.total_loops):
    if args.total_loops > 1:
        print(f"\n{'='*50}")
        print(f" Avaliando modelo {loop_idx + 1}/{args.total_loops}")
        print(f"{'='*50}\n")

    model = get_model(args.model, num_classes)
    model_path = os.path.join("saved_models", args.dataset, f"{args.model}_{loop_idx+1}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    for eps in all_eps:
        acc, prec, rec, f1 = accuracy_FGSM(model, test_loader, eps, device, args.normalize)
        
    for eps in all_eps:
        acc, prec, rec, f1 = accuracy_PGD(model, test_loader, eps, device, args.normalize)

    for eps in all_eps:
        acc, prec, rec, f1 = accuracy_MIM(model, test_loader, eps, device, args.normalize)
        
    if not args.ignore_autoattack:
        for eps in all_eps:
            acc, prec, rec, f1 = accuracy_AutoAttack(model, test_loader, num_classes, eps, device, args.normalize)