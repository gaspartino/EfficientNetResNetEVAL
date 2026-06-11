import torch
import torchattacks
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataset_loader, attack=None, attack_name="Clean", device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true_all, y_pred_all = [], []

    for i, (x, y) in enumerate(dataset_loader):
        x, y = x.to(device), y.to(device)

        if attack is not None:
            x = attack(x, y)

        with torch.no_grad():
            outputs = model(x)
            predicted_class = outputs.argmax(dim=1)

        y_true_all.extend(y.cpu().numpy())
        y_pred_all.extend(predicted_class.cpu().numpy())

    acc = (torch.tensor(y_true_all) == torch.tensor(y_pred_all)).float().mean().item()
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print(f"{attack_name} | Accuracy:  {acc * 100:.2f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1-score: {f1:.4f}%")

    return acc, precision, recall, f1

def accuracy_clean(model, dataset_loader, device):
    return evaluate_model(model, dataset_loader, None, "Clean", device)

def accuracy_FGSM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.FGSM(model, eps=eps)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"FGSM (ε={eps:.2f})", device)

def accuracy_PGD(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.PGD(model, eps=eps)   
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"PGD (ε={eps:.2f})", device)

def accuracy_MIM(model, dataset_loader, eps, device, normalize):
    attack = torchattacks.MIFGSM(model, eps=eps) 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"MIM (ε={eps:.2f})", device)
    
def accuracy_AutoAttack(model, dataset_loader, num_classes, eps, device, normalize):
    attack = torchattacks.AutoAttack(model, eps=eps, n_classes=num_classes)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if normalize:
        attack.set_normalization_used(mean=mean, std=std)

    return evaluate_model(model, dataset_loader, attack, f"AutoAttack (ε={eps:.2f})", device)
