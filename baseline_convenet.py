# Baseline ConvNeXt Classifier for Ablation Study
#
# This script trains a standard ConvNeXt-Tiny model with two separate classification
# heads for the disease and severity tasks. It serves as a strong baseline to
# compare against the more complex TSTC-v2 architecture.
#
# Key Features of this Baseline:
# 1.  Standard Architecture: A single, pre-trained ConvNeXt backbone with its global
#     average pooling layer. No branching, no deep supervision, no hierarchical fusion.
# 2.  Dual Heads: The pooled feature vector from the backbone is fed into two
#     independent linear layers for the two tasks.
# 3.  Robust Training: It uses class-balanced loss and differential learning rates
#     to ensure a fair and strong comparison.
# 4.  Logging and Configuration: Includes file logging and configurable output paths
#     for easy use in automated or containerized environments like Docker.
# 5.  Constant LR: This version uses a constant learning rate with no scheduler or warm-up.

# --- Core Imports ---
import os
import json
import copy
import random
import logging
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import timm

# ==============================================================================
# 1. BASELINE MODEL ARCHITECTURE
# ==============================================================================

class ConvNeXtBaseline(nn.Module):
    """
    A standard ConvNeXt model with a shared backbone and two separate heads.
    """
    def __init__(self, num_disease_classes: int, num_severity_classes: int, model_name='convnext_tiny'):
        super().__init__()
        # Load a pretrained backbone, setting num_classes=0 removes the final classifier
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_backbone_features = self.backbone.num_features

        # Define two separate linear heads for each task
        self.fc_disease = nn.Linear(num_backbone_features, num_disease_classes)
        self.fc_severity = nn.Linear(num_backbone_features, num_severity_classes)

    def get_parameter_groups(self):
        """Helper function to group parameters for differential learning rate."""
        backbone_params = self.backbone.parameters()
        head_params = list(self.fc_disease.parameters()) + list(self.fc_severity.parameters())
        return backbone_params, head_params

    def forward(self, x: torch.Tensor):
        # The timm model with num_classes=0 automatically applies pooling
        features = self.backbone(x) # Shape: (batch_size, num_features)
        
        # Pass the shared features to each head
        disease_out = self.fc_disease(features)
        severity_out = self.fc_severity(features)
        
        return disease_out, severity_out

# ==============================================================================
# 2. BASELINE LOSS FUNCTION
# ==============================================================================

class BaselineLoss(nn.Module):
    """
    A simple loss function that is the sum of the two task losses.
    """
    def __init__(self, disease_weight: torch.Tensor = None, severity_weight: torch.Tensor = None):
        super().__init__()
        self.register_buffer("w_disease", disease_weight)
        self.register_buffer("w_severity", severity_weight)

    def forward(self, outputs, targets):
        disease_out, severity_out = outputs
        disease_labels, severity_labels = targets
        
        # Calculate weighted cross-entropy for each task
        disease_loss = F.cross_entropy(disease_out, disease_labels, weight=self.w_disease)
        severity_loss = F.cross_entropy(severity_out, severity_labels, weight=self.w_severity)
        
        # The total loss is their sum
        return disease_loss + severity_loss

# ==============================================================================
# 3. DATA HANDLING (Unchanged)
# ==============================================================================

SELECTED_LABEL_IDS = [
    14,15, 39,40, 48,49, 52,53, 50,51, 34,35,44,45, 36,37,46,47, 10,11, 28,29,
    7,8,42,43, 54,55, 22,23, 20,21, 12,13, 4,5, 56,57, 26,27, 2,3, 18,19,
    0,6,9,17,24,30,33,38,41
]

class AIChallengerSubset(Dataset):
    def __init__(self, json_path: str, img_root: str, transform=None):
        self.img_root = img_root
        self.transform = transform
        with open(json_path, 'r') as f: all_data = json.load(f)
        self.samples = [(item["image_id"], item["disease_class"]) for item in all_data if item["disease_class"] in SELECTED_LABEL_IDS]
        self.label2disease, self.label2severity = self._build_label_maps()
        self.num_disease_classes = len(set(self.label2disease.values()))
        self.num_severity_classes = len(set(self.label2severity.values()))

    def _build_label_maps(self):
        disease_groups = [[14,15],[39,40],[48,49],[52,53],[50,51],[34,35,44,45],[36,37,46,47],[10,11],[28,29],[7,8,42,43],[54,55],[22,23],[20,21],[12,13],[4,5],[56,57],[26,27],[2,3],[18,19]]
        label2disease, label2severity = {}, {}
        for i, grp in enumerate(disease_groups, 1):
            for label in grp:
                label2disease[label] = i
                label2severity[label] = 2 if (label % 2 == 1) else 1
        for label in [0, 6, 9, 17, 24, 30, 33, 38, 41]:
            label2disease[label] = 0
            label2severity[label] = 0
        return label2disease, label2severity

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_name, numeric_label = self.samples[idx]
        img_path = os.path.join(self.img_root, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.label2disease[numeric_label], self.label2severity[numeric_label]

def get_loaders(train_json, train_images, val_json, val_images, batch_size):
    tf_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_ds = AIChallengerSubset(train_json, train_images, transform=tf_train)
    val_ds = AIChallengerSubset(val_json, val_images, transform=tf_eval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# ==============================================================================
# 4. CLASS WEIGHTING & EVALUATION (Unchanged)
# ==============================================================================

def _compute_label_counts(train_ds):
    d_counts = torch.zeros(train_ds.num_disease_classes, dtype=torch.long)
    s_counts = torch.zeros(train_ds.num_severity_classes, dtype=torch.long)
    for _, numeric_label in train_ds.samples:
        d = train_ds.label2disease[numeric_label]
        s = train_ds.label2severity[numeric_label]
        d_counts[d] += 1
        s_counts[s] += 1
    return d_counts, s_counts

def _class_weights_from_counts(counts, beta=0.999):
    effective_num = 1.0 - torch.pow(beta, counts.float())
    weights = (1.0 - beta) / effective_num
    return weights / torch.sum(weights) * len(counts)

@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    all_d_labels, all_s_labels, all_d_preds, all_s_preds = [], [], [], []
    for images, d_labels, s_labels in loader:
        images, d_labels, s_labels = images.to(device), d_labels.to(device), s_labels.to(device)
        d_logits, s_logits = model(images)
        all_d_labels.append(d_labels.cpu().numpy())
        all_s_labels.append(s_labels.cpu().numpy())
        all_d_preds.append(d_logits.argmax(1).cpu().numpy())
        all_s_preds.append(s_logits.argmax(1).cpu().numpy())
    d_acc = accuracy_score(np.concatenate(all_d_labels), np.concatenate(all_d_preds))
    s_acc = accuracy_score(np.concatenate(all_s_labels), np.concatenate(all_s_preds))
    return d_acc * 100.0, s_acc * 100.0

# ==============================================================================
# 5. BASELINE TRAINING FUNCTION (with Constant LR)
# ==============================================================================

def train_baseline_model(config, model_save_path: str):
    """
    Trains the baseline model with a constant learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    
    TRAIN_JSON = "/data/classification/ai_challenger_pdr2018/train/train_label.json"
    TRAIN_IMG_DIR = "/data/classification/ai_challenger_pdr2018/train/images"
    VAL_JSON = "/data/classification/ai_challenger_pdr2018/test/test_label.json" 
    VAL_IMG_DIR = "/data/classification/ai_challenger_pdr2018/test/images"
    
    
    train_loader, val_loader = get_loaders(TRAIN_JSON, TRAIN_IMG_DIR, VAL_JSON, VAL_IMG_DIR, config['batch_size'])

    num_disease = train_loader.dataset.num_disease_classes
    num_severity = train_loader.dataset.num_severity_classes

    d_counts, s_counts = _compute_label_counts(train_loader.dataset)
    d_weights = _class_weights_from_counts(d_counts).to(device)
    s_weights = _class_weights_from_counts(s_counts).to(device)
    
    model = ConvNeXtBaseline(num_disease, num_severity).to(device)
    criterion = BaselineLoss(disease_weight=d_weights, severity_weight=s_weights)
    
    backbone_params, head_params = model.get_parameter_groups()
    optimizer = optim.AdamW([
        {'params': backbone_params},
        {'params': head_params}
    ], lr=config['lr'])
    
    # --- REMOVED: LR Scheduler and Warm-up for constant LR ---
    
    best_combined_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}", leave=True)
        for images, d_labels, s_labels in progress_bar:
            images, d_labels, s_labels = images.to(device), d_labels.to(device), s_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, (d_labels, s_labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[1]['lr']:.6f}")

        val_d_acc, val_s_acc = evaluate(val_loader, model, device)
        combined_acc = 0.5 * val_d_acc + 0.5 * val_s_acc
        
        logging.info(f"Epoch {epoch}: Val Disease Acc: {val_d_acc:.2f}%, Val Severity Acc: {val_s_acc:.2f}%, Combined: {combined_acc:.2f}%")
        print(f"Epoch {epoch}: Val Disease Acc: {val_d_acc:.2f}%, Val Severity Acc: {val_s_acc:.2f}%, Combined: {combined_acc:.2f}%")

        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            logging.info(f"  -> New best score! Saving model to '{model_save_path}'")
            torch.save(model.state_dict(), model_save_path)

# ==============================================================================
# 6. SETUP AND MAIN EXECUTION
# ==============================================================================

def setup_logging(log_dir: str):
    log_file = os.path.join(log_dir, 'baseline_training_log_constant_lr.txt') # New log file name
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    
    # --- Configurable Output Directories ---
    OUTPUT_DIR = "/data/classification/outputs_convenet_base/"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    setup_logging(LOG_DIR)

    baseline_model_path = os.path.join(MODEL_SAVE_DIR, "baseline_model_constant_lr.pth") # New model save name

    # Use the same hyperparameter family for a fair comparison
    baseline_params = {
        'lr': 0.0001,
        'weight_decay': 0.0002591,
        'epochs': 50,
        'batch_size': 32
    }

    logging.info("--- Starting Baseline Model Training Run (Constant LR) ---")
    logging.info(f"Parameters: {baseline_params}")
    logging.info(f"Model checkpoints will be saved to: {baseline_model_path}")
    logging.info(f"Logs will be saved to: {os.path.join(LOG_DIR, 'baseline_training_log_constant_lr.txt')}")
    
    train_baseline_model(baseline_params, model_save_path=baseline_model_path)
    
    logging.info("\n--- Baseline Training Complete ---")
    logging.info(f"The best baseline model has been saved to '{baseline_model_path}'")

