#this is with new code, the weight decay is removed, and incorporated a change with DFR Architecture 
#also we have added more augmentations 
# Final Plant Disease & Severity Classifier Training (DFR Version)
#
# This script implements a novel architecture for improved performance and publication.
#
# Key Architectural Changes:
# 1.  Simplified Shared Backbone: Replaces the redundant multi-branch design with a
#     single, efficient ConvNeXt backbone for powerful shared feature extraction.
# 2.  Dynamic Feature Reweighting (DFR) Module: A novel component where the disease
#     classification task explicitly guides the feature selection for the severity task.
#     This mimics an expert's workflow and creates a strong, publishable contribution.
#
# This script retains the robust training techniques from the previous version:
# - Differential Learning Rates
# - Learning Rate Warm-up & Cosine Annealing Scheduler
# - Gradient Clipping
# - Weighted Cross-Entropy Loss for class imbalance
# - Configurable Paths & Comprehensive Logging

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
# 1. ADVANCED MODEL ARCHITECTURE (Shared Backbone + DFR Module)
# ==============================================================================

# NEW: The novel module for task-guided attention.
class DFR_Module(nn.Module):
    """
    Dynamic Feature Reweighting Module.
    This module uses the disease prediction logits to generate a "guidance vector"
    that re-weights the shared features, making them more informative for the
    severity prediction task.
    """
    def __init__(self, num_disease_classes: int, feature_channels: int):
        super().__init__()
        # This small network learns to create the guidance vector.
        self.gating_network = nn.Sequential(
            nn.Linear(num_disease_classes, feature_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // 4, feature_channels),
            nn.Sigmoid()  # Use Sigmoid to scale weights between 0 and 1
        )

    def forward(self, shared_features: torch.Tensor, disease_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features (torch.Tensor): The feature vector from the shared backbone.
            disease_logits (torch.Tensor): The output from the disease classification head.

        Returns:
            torch.Tensor: The re-weighted feature vector for the severity task.
        """
        # Generate the guidance vector from the disease logits
        guidance_vector = self.gating_network(disease_logits)
        # Multiply each feature channel by its corresponding learned weight
        reweighted_features = shared_features * guidance_vector
        return reweighted_features

# MODIFIED: The main model is now simplified and incorporates the DFR module.
class TSTC_v2_Simplified(nn.Module):
    def __init__(self, num_disease_classes: int, num_severity_classes: int):
        super().__init__()
        # --- 1. SHARED BACKBONE ---
        # Use features_only=True for easy access to intermediate feature maps
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, features_only=True)
        # Get the number of channels from the final backbone stage
        backbone_channels = self.backbone.feature_info.channels()
        final_feature_dim = backbone_channels[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- 2. CLASSIFICATION HEADS ---
        self.fc_disease = nn.Linear(final_feature_dim, num_disease_classes)
        self.fc_severity_main = nn.Linear(final_feature_dim, num_severity_classes)
        self.fc_severity_aux = nn.Linear(final_feature_dim, num_severity_classes)

        # --- 3. THE NOVEL DFR MODULE ---
        self.dfr = DFR_Module(num_disease_classes=num_disease_classes, feature_channels=final_feature_dim)

    def get_parameter_groups(self):
        """Helper function to group parameters for differential learning rate."""
        backbone_params = list(self.backbone.parameters())
        head_params = (list(self.pool.parameters()) +
                       list(self.fc_disease.parameters()) +
                       list(self.fc_severity_main.parameters()) +
                       list(self.fc_severity_aux.parameters()) +
                       list(self.dfr.parameters()))
        return backbone_params, head_params

    def forward(self, x: torch.Tensor):
        # --- 1. Get Shared Features ---
        feature_maps = self.backbone(x)
        final_feature_map = feature_maps[-1]  # Get the deepest feature map
        shared_vector = self.pool(final_feature_map).flatten(1)

        # --- 2. Disease Prediction ---
        disease_logits = self.fc_disease(shared_vector)

        # --- 3. Dynamic Reweighting (The novel step!) ---
        # The disease prediction now GUIDES the feature selection for severity.
        severity_features_reweighted = self.dfr(shared_vector, disease_logits)

        # --- 4. Severity Prediction ---
        sev_out_main = self.fc_severity_main(severity_features_reweighted)
        
        # The auxiliary head can work on the original features for regularization
        sev_out_aux = self.fc_severity_aux(shared_vector)

        if self.training:
            return disease_logits, sev_out_main, sev_out_aux
        else:
            # For evaluation, we only need the main predictions
            return disease_logits, sev_out_main

# ==============================================================================
# 2. OPTIMIZED LOSS FUNCTION (Unchanged)
# ==============================================================================

class OptimizedLoss(nn.Module):
    def __init__(self, disease_weight=None, severity_weight=None, aux_lambda=0.5):
        super().__init__()
        self.aux_lambda = aux_lambda
        self.register_buffer("w_disease", disease_weight)
        self.register_buffer("w_severity", severity_weight)

    def forward(self, outputs, targets):
        disease_out, sev_out_main, sev_out_aux = outputs
        disease_labels, severity_labels = targets
        l1_disease = F.cross_entropy(disease_out, disease_labels, weight=self.w_disease)
        l2_sev_main = F.cross_entropy(sev_out_main, severity_labels, weight=self.w_severity)
        l3_sev_aux = F.cross_entropy(sev_out_aux, severity_labels, weight=self.w_severity)
        return l1_disease + l2_sev_main + self.aux_lambda * l3_sev_aux

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
    tf_train = transforms.Compose([
      transforms.Resize((224, 224)),
      
      # 1. Use an automated augmentation policy. TrivialAugment is state-of-the-art.
      transforms.TrivialAugmentWide(),
      
      # 2. Convert image to tensor BEFORE random erasing
      transforms.ToTensor(),
      
      # 3. Apply normalization
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      
      # 4. Apply Random Erasing on the normalized tensor
      # p=0.25 means it will be applied to 25% of the images in a batch.
      transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
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

# MODIFIED: Replace the old function with this one.
def _class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """
    Computes class weights using the simple and effective Inverse Class Frequency method.
    """
    # Add a small epsilon to avoid division by zero if a class has 0 samples
    weights = 1.0 / (counts.float() + 1e-8)
    
    # Normalize weights so that they sum to the number of classes
    normalized_weights = weights / torch.sum(weights) * len(counts)
    
    return normalized_weights

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
# 5. STABLE FINAL TRAINING FUNCTION (Now using the new model)
# ==============================================================================

def train_final_model_dfr(config, model_save_path: str):
    """
    Trains the final model with the DFR module and saves to a specified path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # These paths should be configured for your environment
    TRAIN_JSON = "/data/classification/ai_challenger_pdr2018/train/train_label.json"
    TRAIN_IMG_DIR = "/data/classification/ai_challenger_pdr2018/train/images"
    VAL_JSON = "/data/classification/ai_challenger_pdr2018/test/test_label.json" 
    VAL_IMG_DIR = "/data/classification/ai_challenger_pdr2018/test/images"
    
    train_loader, val_loader = get_loaders(TRAIN_JSON, TRAIN_IMG_DIR, VAL_JSON, VAL_IMG_DIR, config['batch_size'])

    num_disease = train_loader.dataset.num_disease_classes
    num_severity = train_loader.dataset.num_severity_classes

    # MODIFIED: Using Weighted Cross-Entropy as decided.
    logging.info("Calculating class weights for Weighted Cross-Entropy...")
    d_counts, s_counts = _compute_label_counts(train_loader.dataset)
    d_weights = _class_weights_from_counts(d_counts).to(device)
    s_weights = _class_weights_from_counts(s_counts).to(device)
    logging.info(f"Severity weights: {s_weights.cpu().numpy().round(2)}")
    
    # MODIFIED: Instantiate the new simplified model
    model = TSTC_v2_Simplified(num_disease, num_severity).to(device)
    criterion = OptimizedLoss(disease_weight=d_weights, severity_weight=s_weights, aux_lambda=config['aux_lambda'])
    
    backbone_params, head_params = model.get_parameter_groups()
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},
        {'params': head_params, 'lr': config['lr']}
    ])
    
    warmup_epochs = 5
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs)
    
    best_combined_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.0
        
        # Manual warm-up logic
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for i, param_group in enumerate(optimizer.param_groups):
                base_lr = config['lr'] * 0.1 if i == 0 else config['lr']
                param_group['lr'] = base_lr * warmup_factor
        
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
        
        if epoch > warmup_epochs:
            main_scheduler.step()
        
        logging.info(f"Epoch {epoch}: Val Disease Acc: {val_d_acc:.2f}%, Val Severity Acc: {val_s_acc:.2f}%, Combined: {combined_acc:.2f}%")

        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            logging.info(f"  -> New best score! Saving model to '{model_save_path}'")
            torch.save(model.state_dict(), model_save_path)

# ==============================================================================
# 6. SETUP AND MAIN EXECUTION (Unchanged)
# ==============================================================================

def setup_logging(log_dir: str):
    """Configures logging to file and console."""
    log_file = os.path.join(log_dir, 'training_log_dfr_aug_1.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    
    # --- Configurable Output Directories ---
    OUTPUT_DIR = "/data/classification/outputs_tstc_dfr_arch_aug/"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    setup_logging(LOG_DIR)

    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_dfr.pth")

    # Using the best hyperparameters found from the Optuna search as a starting point
    best_params = {
        'lr': 0.0001,
        'weight_decay': 0.0002591,
        'aux_lambda': 0.91533,
        'epochs': 50,
        'batch_size': 32
    }

    logging.info("--- Starting Final Training Run with DFR Architecture ---")
    logging.info(f"Parameters: {best_params}")
    logging.info(f"Model will be saved to: {best_model_path}")
    logging.info(f"Logs will be saved to: {os.path.join(LOG_DIR, 'training_log_dfr.txt')}")
    
    train_final_model_dfr(best_params, model_save_path=best_model_path)
    
    logging.info("\n--- Final DFR Training Complete ---")
    logging.info(f"The best model has been saved to '{best_model_path}'")