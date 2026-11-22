"""FC loading and the original MUSA MLP classifier."""

import os

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def set_global_seed(random_state: int):
    # Set Python/numpy/torch/cuda random seeds for reproducibility
    import random
    
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        uda = getattr(torch, "use_deterministic_algorithms", None)
        if uda is not None:
            uda(True, warn_only=True)

class MLPClassifier(nn.Module):
    # Fully connected layers + BN + ReLU + Dropout
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def flatten_and_classify_mlp(features, labels, train_indices, test_indices, method_name="", 
                             hidden_dims=[640, 512, 384, 256], dropout=0.25, batch_size=64, 
                             num_epochs=400, learning_rate=0.0009, device=None,
                             val_split=0.15, val_indices=None, early_stop_patience=100, early_stop_min_delta=1e-4,
                             enable_early_stop=True, checkpoint_interval=10, use_multi_checkpoint=False):
    # Flatten multi-ROI features -> StandardScaler -> AdamW MLP training, return test metrics
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = features.shape[0]
    features_flat = features.reshape(N, -1)
    input_dim = features_flat.shape[1]
    
    scaler = StandardScaler()
    scaler.fit(features_flat[train_indices])
    features_normalized = scaler.transform(features_flat)
    
    if val_indices is not None:
        train_idx = np.array(train_indices)
        val_idx = np.array(val_indices)
    else:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            np.array(train_indices),
            test_size=val_split,
            stratify=labels[train_indices],
            random_state=42,
            shuffle=True,
        )
    
    X_train = torch.FloatTensor(features_normalized[train_idx]).to(device)
    y_train = torch.LongTensor(labels[train_idx]).to(device)
    X_val = torch.FloatTensor(features_normalized[val_idx]).to(device)
    y_val = torch.LongTensor(labels[val_idx]).to(device)
    X_test = torch.FloatTensor(features_normalized[test_indices]).to(device)
    y_test = labels[test_indices]
    
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = 15
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    best_checkpoint = None
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_val = val_loss.item()
            val_pred = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_pred)
        
        improved = val_loss_val < best_val_loss - early_stop_min_delta
        if improved:
            best_val_loss = val_loss_val
                                    patience_counter = 0
            best_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss_val,
                'val_acc': val_acc,
            }
        else:
            patience_counter += 1
        
        if use_multi_checkpoint and (epoch + 1) % checkpoint_interval == 0:
            if best_checkpoint is None or val_acc > best_checkpoint["val_acc"]:
                best_checkpoint = {
                    "model": {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()},
                    "val_acc": val_acc,
                }
            model.train()
        
        if enable_early_stop and patience_counter >= early_stop_patience:
            break
        
        model.train()
    
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["model"])
    elif best_state is not None:
        model.load_state_dict(best_state["model"])

    model.eval()
    with torch.no_grad():
        predicted_labels = model(X_test).argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y_test, predicted_labels))
