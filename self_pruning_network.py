"""
Self-Pruning Neural Network for CIFAR-10
=========================================
Tredence Analytics – AI Engineer Case Study

This module implements a feed-forward neural network with learnable gate parameters
that enable dynamic, self-directed weight pruning during training.

Key concepts:
  • PrunableLinear layer: each weight w_ij has a learnable gate score g_ij.
    The effective weight is  w_ij * sigmoid(g_ij).
  • Sparsity loss: L1 norm of all gate values drives them towards 0.
  • Total Loss = CrossEntropyLoss + λ * Σ|sigmoid(g_ij)|


"""

import os
import math
import time
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear Layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that attaches a learnable gate score
    to every weight.

    Forward pass
    ────────────
        gates        = sigmoid(gate_scores)          ∈ (0, 1)^{out × in}
        pruned_w     = weight ⊙ gates
        output       = input @ pruned_w.T + bias

    Gradients flow through both `weight` and `gate_scores` because
    all operations (sigmoid, element-wise mul, matmul) are differentiable.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard weight & bias ──────────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Learnable gate scores ───────────────────────────────────────────
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Kaiming uniform for weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Gate scores initialization: 2.0 → sigmoid ≈ 0.88, gates start mostly open
        nn.init.constant_(self.gate_scores, 2.0)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.gate_values()
        return (gates < threshold).float().mean().item()

# ──────────────────────────────────────────────────────────────────────────────
# 1.1 PrunableConv2d Layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableConv2d(nn.Module):
    """
    A drop-in replacement for nn.Conv2d with learnable gate parameters.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.gate_scores = nn.Parameter(torch.empty_like(self.weight))
        self.stride  = stride
        self.padding = padding
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize gate scores to 2.0 (sigmoid ≈ 0.88) so gates start mostly open
        nn.init.constant_(self.gate_scores, 2.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.conv2d(x, pruned_weight, self.bias, stride=self.stride, padding=self.padding)

    @torch.no_grad()
    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Network Definition
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    CNN architecture for CIFAR-10 with self-pruning gates.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            PrunableConv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            PrunableConv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            PrunableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            PrunableLinear(256, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _get_prunable_layers(self):
        layers = []
        for m in self.modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                layers.append(m)
        return layers

    def all_gates(self) -> Tensor:
        return torch.cat([l.gate_values().view(-1) for l in self._get_prunable_layers()])

    def sparsity_loss(self) -> Tensor:
        # Sum of all gate values (L1 norm) as specified in Part 2 of the Case Study.
        # This encourages the optimizer to drive individual gate values to exactly zero.
        total_sp_loss = 0
        for l in self._get_prunable_layers():
            total_sp_loss += torch.sigmoid(l.gate_scores).sum()
        return total_sp_loss

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()

    @torch.no_grad()
    def total_weights(self) -> int:
        return sum(l.weight.numel() for l in self._get_prunable_layers())

    @torch.no_grad()
    def active_weights(self, threshold: float = 1e-2) -> int:
        total = 0
        for l in self._get_prunable_layers():
            total += (l.gate_values() >= threshold).sum().item()
        return int(total)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    batch_size: int = 256,
    num_workers: int = 0, # Set to 0 for better stability in Windows env
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: optim.lr_scheduler._LRScheduler = None,
) -> Tuple[float, float, float]:
    model.train()
    total_loss_sum = ce_loss_sum = sp_loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits      = model(images)
            ce_loss     = criterion(logits, labels)
            sp_loss     = model.sparsity_loss()
            total_loss  = ce_loss + lam * sp_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss_sum += total_loss.item()
        ce_loss_sum    += ce_loss.item()
        sp_loss_sum    += sp_loss.item()

    n = len(loader)
    return total_loss_sum / n, ce_loss_sum / n, sp_loss_sum / n

@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total

def train(
    lam: float,
    epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    lr: float = 3e-4,
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(42)
    model = SelfPruningNet(dropout=0.3).to(device)

    # weight decay only on weights
    weight_params = [p for n, p in model.named_parameters()
                     if "gate_scores" not in n and "bias" not in n]
    gate_params   = [p for n, p in model.named_parameters()
                     if "gate_scores" in n]
    bias_params   = [p for n, p in model.named_parameters()
                     if "bias" in n]

    optimizer = optim.AdamW([
        {"params": weight_params, "weight_decay": 0.05},
        {"params": gate_params,   "weight_decay": 0.0},
        {"params": bias_params,   "weight_decay": 0.0},
    ], lr=lr)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*10, steps_per_epoch=len(train_loader), epochs=epochs)
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    history = {
        "total_loss": [], "ce_loss": [], "sp_loss": [],
        "sparsity":   [], "test_acc": [],
    }

    if verbose:
        print(f"\nTraining lambda = {lam:.4f}")

    for epoch in range(1, epochs + 1):
        t_loss, c_loss, s_loss = train_one_epoch(
            model, train_loader, optimizer, lam, device, scaler, scheduler)

        sparsity = model.global_sparsity()
        test_acc = evaluate(model, test_loader, device)

        history["total_loss"].append(t_loss)
        history["ce_loss"].append(c_loss)
        history["sp_loss"].append(s_loss)
        history["sparsity"].append(sparsity * 100)
        history["test_acc"].append(test_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:02d} | Loss: {t_loss:.4f} | Sparsity: {sparsity*100:.1f}% | Acc: {test_acc:.2f}%")

    return {
        "lam":           lam,
        "model":         model,
        "history":       history,
        "final_acc":     test_acc,
        "final_sparsity": history["sparsity"][-1],
        "gate_values":   model.all_gates().cpu().numpy(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(gate_values: np.ndarray, lam: float, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=60, color="#58a6ff", alpha=0.8)
    plt.title(f"Gate Distribution (λ = {lam})")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def plot_training_curves(results: List[Dict], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res in results:
        lam = res["lam"]
        hist = res["history"]
        axes[0].plot(hist["test_acc"], label=f"λ={lam}")
        axes[1].plot(hist["sparsity"], label=f"λ={lam}")
    
    axes[0].set_title("Test Accuracy")
    axes[1].set_title("Sparsity (%)")
    axes[0].legend()
    axes[1].legend()
    plt.savefig(out_path)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lambdas",    type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    parser.add_argument("--out-dir",    type=str,   default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    train_loader, test_loader = get_cifar10_loaders(batch_size=256, data_dir="./data")

    all_results = []
    for lam in args.lambdas:
        res = train(lam, args.epochs, device, train_loader, test_loader)
        all_results.append(res)
        plot_gate_distribution(res["gate_values"], lam, os.path.join(args.out_dir, f"gates_lam_{lam}.png"))

    plot_training_curves(all_results, os.path.join(args.out_dir, "training.png"))

    print("\nSummary Table:")
    print("| Lambda | Accuracy | Sparsity |")
    print("|---|---|---|")
    for r in all_results:
        print(f"| {r['lam']} | {r['final_acc']:.2f}% | {r['final_sparsity']:.1f}% |")

if __name__ == "__main__":
    main()
