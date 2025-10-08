# Final Plant Disease & Severity Classifier Training (Stable Version with Logging)
#
# This script addresses training instability observed in the disease classification head.
# It introduces several key techniques for more robust and stable training:
#
# 1.  Differential Learning Rates: The pre-trained backbone gets a smaller learning rate
#     for fine-tuning, while new heads get a larger learning rate.
# 2.  Learning Rate Warm-up & Scheduler: Training starts with a low LR, warms up,
#     and then gradually decreases for stable convergence.
# 3.  Gradient Clipping: Prevents exploding gradients that can cause accuracy drops.
# 4.  Configurable Paths & Logging: All outputs (model checkpoints and logs) are saved
#     to explicitly defined directories, making it ideal for Docker and automated runs.

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
# 1. ADVANCED MODEL ARCHITECTURE (Now with parameter grouping)
# ==============================================================================

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HierarchicalAttentionFusion(nn.Module):
    def __init__(self, stage_channels: List[int], common_dim: int = 256):
        super().__init__()
        self.proj_s2 = nn.Conv2d(stage_channels[0], common_dim, kernel_size=1)
        self.proj_s3 = nn.Conv2d(stage_channels[1], common_dim, kernel_size=1)
        self.proj_s4 = nn.Conv2d(stage_channels[2], common_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(common_dim * 3)
        self.attention = SqueezeExcite(common_dim * 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = common_dim * 3

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        s2_feat, s3_feat, s4_feat = features
        s2_proj = self.proj_s2(s2_feat)
        s3_proj = self.proj_s3(s3_feat)
        s4_proj = self.proj_s4(s4_feat)
        target_size = s2_proj.shape[2:]
        s3_upsampled = F.interpolate(s3_proj, size=target_size, mode='bilinear', align_corners=False)
        s4_upsampled = F.interpolate(s4_proj, size=target_size, mode='bilinear', align_corners=False)
        fused_features = torch.cat([s2_proj, s3_upsampled, s4_upsampled], dim=1)
        fused_features = self.bn(fused_features)
        attended_features = self.attention(fused_features)
        pooled_vector = self.pool(attended_features).flatten(1)
        return pooled_vector

class TSTC_v2(nn.Module):
    def __init__(self, num_disease_classes: int, num_severity_classes: int):
        super().__init__()
        full_backbone = timm.create_model('convnext_tiny', pretrained=True)
        backbone_channels = [info['num_chs'] for info in full_backbone.feature_info]
        
        # --- Group Backbone Parameters ---
        self.stem = full_backbone.stem
        self.stage0 = full_backbone.stages[0]
        self.stage1 = full_backbone.stages[1]
        self.branch1_s2 = copy.deepcopy(full_backbone.stages[2])
        self.branch1_s3 = copy.deepcopy(full_backbone.stages[3])
        self.branch2_s2 = copy.deepcopy(full_backbone.stages[2])
        self.branch2_s3 = copy.deepcopy(full_backbone.stages[3])
        self.branch3_s2 = copy.deepcopy(full_backbone.stages[2])
        self.branch3_s3 = copy.deepcopy(full_backbone.stages[3])

        # --- Group Head Parameters ---
        self.hierarchical_fusion = HierarchicalAttentionFusion(
            stage_channels=[backbone_channels[1], backbone_channels[2], backbone_channels[3]]
        )
        fusion_dim = self.hierarchical_fusion.output_dim
        self.fc_disease = nn.Linear(fusion_dim, num_disease_classes)
        self.pool_sev_main = nn.AdaptiveAvgPool2d(1)
        self.fc_severity_main = nn.Linear(backbone_channels[3], num_severity_classes)
        self.pool_sev_aux = nn.AdaptiveAvgPool2d(1)
        self.fc_severity_aux = nn.Linear(backbone_channels[3], num_severity_classes)

    def get_parameter_groups(self):
        """Helper function to group parameters for differential learning rate."""
        backbone_params = list(self.stem.parameters()) + \
                          list(self.stage0.parameters()) + \
                          list(self.stage1.parameters()) + \
                          list(self.branch1_s2.parameters()) + \
                          list(self.branch1_s3.parameters()) + \
                          list(self.branch2_s2.parameters()) + \
                          list(self.branch2_s3.parameters()) + \
                          list(self.branch3_s2.parameters()) + \
                          list(self.branch3_s3.parameters())
                          
        head_params = list(self.hierarchical_fusion.parameters()) + \
                      list(self.fc_disease.parameters()) + \
                      list(self.pool_sev_main.parameters()) + \
                      list(self.fc_severity_main.parameters()) + \
                      list(self.pool_sev_aux.parameters()) + \
                      list(self.fc_severity_aux.parameters())
                      
        return backbone_params, head_params

    def forward(self, x: torch.Tensor):
        shared_features = self.stem(x)
        s1_out = self.stage0(shared_features)
        s2_in = self.stage1(s1_out)
        b1_s2_out = self.branch1_s2(s2_in)
        b1_s3_out = self.branch1_s3(b1_s2_out)
        hierarchical_features = self.hierarchical_fusion([s2_in, b1_s2_out, b1_s3_out])
        disease_out = self.fc_disease(hierarchical_features)
        b2_s2_out = self.branch2_s2(s2_in)
        b2_s3_out = self.branch2_s3(b2_s2_out)
        feat_sev_aux = self.pool_sev_aux(b2_s3_out).flatten(1)
        sev_out_aux = self.fc_severity_aux(feat_sev_aux)
        b3_s2_out = self.branch3_s2(s2_in)
        b3_s3_out = self.branch3_s3(b3_s2_out)
        feat_sev_main = self.pool_sev_main(b3_s3_out).flatten(1)
        sev_out_main = self.fc_severity_main(feat_sev_main)
        if self.training:
            return disease_out, sev_out_main, sev_out_aux
        else:
            return disease_out, sev_out_main

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
# 5. STABLE FINAL TRAINING FUNCTION (Now with logging)
# ==============================================================================

def train_final_model_stable(config, model_save_path: str):
    """
    Trains the final model with stability enhancements and saves to a specified path.
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
    
    model = TSTC_v2(num_disease, num_severity).to(device)
    criterion = OptimizedLoss(disease_weight=d_weights, severity_weight=s_weights, aux_lambda=config['aux_lambda'])
    
    backbone_params, head_params = model.get_parameter_groups()
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},
        {'params': head_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])
    
    warmup_epochs = 5
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs)
    
    best_combined_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.0
        
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
# 6. SETUP AND MAIN EXECUTION
# ==============================================================================

def setup_logging(log_dir: str):
    """Configures logging to file and console."""
    log_file = os.path.join(log_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    
    # --- NEW: Configurable Output Directories ---
    # Define a base output directory, e.g., for mounting in Docker
    OUTPUT_DIR = "/data/classification/outputs_tstc/"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- NEW: Setup Logging ---
    setup_logging(LOG_DIR)

    # Define the full path for the best model
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_stable.pth")

    # Using the best hyperparameters found from the Optuna search
    best_params = {
        'lr': 0.00038765,
        'weight_decay': 0.0002591,
        'aux_lambda': 0.91533,
        'epochs': 50,
        'batch_size': 32
    }

    logging.info("--- Starting Final Stable Training Run with Best Hyperparameters ---")
    logging.info(f"Parameters: {best_params}")
    logging.info(f"Model checkpoints will be saved to: {best_model_path}")
    logging.info(f"Logs will be saved to: {os.path.join(LOG_DIR, 'training_log.txt')}")
    
    train_final_model_stable(best_params, model_save_path=best_model_path)
    
    logging.info("\n--- Final Training Complete ---")
    logging.info(f"The best model has been saved to '{best_model_path}'")

