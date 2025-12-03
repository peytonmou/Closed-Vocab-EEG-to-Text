import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from data import *
from models import create_model
from metrics import (Metrics, evaluation, track_best_test_acc, 
                     get_confusion_matrix, plot_confusion_matrix)


@dataclass
class ExperimentResults:
    """Container for experiment results"""
    avg_test_acc: float
    std_test_acc: float
    total_cm: np.ndarray
    class_accs: np.ndarray
    fold_results: Dict[Any, Dict]


def create_save_dir(save_dir: str) -> str:
    """Create timestamped save directory"""
    save_dir = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def create_model_from_args(args: Dict) -> nn.Module:
    model_kwargs = {k: v for k, v in args.items() if k in [
        'time_steps', 'num_electrodes', 'F1', 'F2', 'D', 'kernel_1', 'kernel_2', 
        'dropout', 'momentum', 'k', 'm1', 'm2', 's', 'proj_dim', 'embed_dim',
        'kernel_num', 'kernel_size', 'nhead', 'dim_feedforward', 'num_layers', 'vocab_size'
    ]}
    
    return create_model(
        args['model_name'],
        num_classes=args['class_num'],
        device=args['device'],
        **model_kwargs
    ).to(args['device'])


def train_epoch(model: nn.Module, train_loader, criterion, optimizer, device) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    return train_loss / len(train_loader), 100. * train_correct / train_total


def train(args: Dict, model: nn.Module, train_loader, val_loader, test_loader, 
          class_names: List[str], save_dir: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """Main training loop"""
    save_dir = create_save_dir(save_dir)
    
    # Save config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False, sort_keys=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    
    metrics = Metrics(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", 
                       "test_loss", "test_acc", "best_val_acc", "best_test_acc", "best_test_epoch"])
    best_val_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    device = args['device']
    
    # Training loop
    for epoch in range(args['epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluation(model, val_loader, criterion, device)
        
        # Test
        test_loss, test_acc, best_test_acc, best_test_epoch = track_best_test_acc(
            model, test_loader, criterion, device, best_test_acc, best_test_epoch, epoch+1)
        
        # Update learning rate
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args['epochs']}: Train {train_acc:.2f}% | "
                  f"Val {val_acc:.2f}% | Test {test_acc:.2f}% | Best Test {best_test_acc:.2f}%")
        
        metrics.add_row([epoch+1, lr, train_loss, train_acc, val_loss, val_acc, 
                         test_loss, test_acc, best_val_acc, best_test_acc, best_test_epoch])
        metrics.save_to_csv(os.path.join(save_dir, "metrics.csv"))
    
    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    test_loss, test_acc = evaluation(model, test_loader, criterion, device)
    
    cm, class_accs = get_confusion_matrix(model, test_loader, device, class_names)
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, "cm.png"))
    
    print(f"\nFinal Test Acc: {test_acc:.2f}%")
    # print(f"Best: {class_names[np.argmax(class_accs)]} ({class_accs.max()*100:.2f}%)")
    # print(f"Worst: {class_names[np.argmin(class_accs)]} ({class_accs.min()*100:.2f}%)")
    
    metrics.df.loc[metrics.df.index[-1], 'test_acc'] = test_acc
    metrics.save_to_csv(os.path.join(save_dir, "metrics.csv"))
    
    return test_acc, cm, class_accs


def calculate_class_accuracies(cm: np.ndarray) -> np.ndarray:
    """Calculate per-class accuracies from confusion matrix"""
    class_accs = []
    for i in range(cm.shape[0]):
        true_positives = cm[i, i]
        total_actual = np.sum(cm[i, :])
        class_accs.append(true_positives / total_actual if total_actual > 0 else 0)
    return np.array(class_accs)


def save_experiment_results(save_dir: str, results: ExperimentResults, 
                           class_names: List[str], exp_name: str, args: Dict):
    """Save all experiment results to files"""
    # Confusion matrix
    plot_confusion_matrix(results.total_cm, class_names, 
                         os.path.join(save_dir, f"{exp_name}_confusion_matrix.png"))
    
    # Class results
    class_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': [f"{acc*100:.2f}%" for acc in results.class_accs]
    })
    class_df.to_csv(os.path.join(save_dir, f"{exp_name}_class_results.csv"), index=False)
    
    # Fold results
    fold_df = pd.DataFrame([
        {'Fold': fold, 'Test Accuracy': f"{res['test_acc']:.2f}%"}
        for fold, res in results.fold_results.items()
    ])
    fold_df.to_csv(os.path.join(save_dir, f"{exp_name}_fold_results.csv"), index=False)
    
    # Summary
    with open(os.path.join(save_dir, f"{exp_name}_summary.txt"), "w") as f:
        f.write(f"{exp_name.upper()} Results - {args['model_name'].upper()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Average Test Accuracy: {results.avg_test_acc:.2f}% ± {results.std_test_acc:.2f}%\n\n")
        f.write(f"Fold-wise Results:\n")
        for fold, res in results.fold_results.items():
            f.write(f"  {fold}: {res['test_acc']:.2f}%\n")
        f.write(f"\nClass-wise Results:\n")
        for name, acc in zip(class_names, results.class_accs):
            f.write(f"  {name}: {acc*100:.2f}%\n")


def run_experiment(args: Dict, X, Y, subjects, class_names: List[str], 
                   data_loader_fn, exp_name: str, fold_iterator) -> ExperimentResults:
    """Generic experiment runner for all cross-validation types"""
    print(f"\n{'='*60}")
    print(f"{exp_name.upper()} - {args['model_name'].upper()}")
    print(f"{'='*60}")
    
    main_save_dir = create_save_dir(os.path.join(args['save_dir'], exp_name.lower().replace(' ', '_')))
    
    all_test_accs = []
    all_cms = []
    fold_results = {}
    
    for fold_id, fold_data in fold_iterator:
        print(f"\n{exp_name} Fold {fold_id}")
        
        # Prepare data
        train_loader, val_loader, test_loader = data_loader_fn(X, Y, subjects, fold_data, args)
        
        # Train
        fold_save_dir = os.path.join(main_save_dir, f"fold_{fold_id}")
        model = create_model_from_args(args)
        test_acc, cm, class_accs = train(args, model, train_loader, val_loader, 
                                         test_loader, class_names, fold_save_dir)
        
        # Store results
        all_test_accs.append(test_acc)
        all_cms.append(cm)
        fold_results[fold_id] = {
            'test_acc': test_acc,
            'cm': cm,
            'class_accuracies': class_accs
        }
    
    # Aggregate results
    total_cm = np.sum(all_cms, axis=0)
    class_accs = calculate_class_accuracies(total_cm)
    
    results = ExperimentResults(
        avg_test_acc=np.mean(all_test_accs),
        std_test_acc=np.std(all_test_accs),
        total_cm=total_cm,
        class_accs=class_accs,
        fold_results=fold_results
    )
    
    # Save results
    save_experiment_results(main_save_dir, results, class_names, exp_name, args)
    
    print(f"\n{exp_name} Average: {results.avg_test_acc:.2f}% ± {results.std_test_acc:.2f}%")
    return results


def run_within_subject_evaluation(args: Dict, X, Y, subjects, class_names: List[str]) -> ExperimentResults:
    """Within-subject evaluation"""
    unique_subjects = np.unique(subjects)
    
    def fold_iterator():
        for i, subject in enumerate(unique_subjects):
            yield subject, subject
    
    return run_experiment(
        args, X, Y, subjects, class_names,
        prepare_data_loaders_within_subject,
        "Within-Subject Evaluation",
        fold_iterator()
    )


def run_cross_validation(args: Dict, X, Y, subjects, class_names: List[str], n_folds: int = 5) -> ExperimentResults:
    """K-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def fold_iterator():
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y)):
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.125, random_state=42, stratify=Y[train_val_idx]
            )
            yield fold+1, (train_idx, val_idx, test_idx)
    
    def data_loader_wrapper(X, Y, subjects, fold_data, args):
        train_idx, val_idx, test_idx = fold_data
        return prepare_data_loaders_cross_validation(X, Y, subjects, train_idx, val_idx, test_idx, args)
    
    return run_experiment(
        args, X, Y, subjects, class_names,
        data_loader_wrapper,
        f"{n_folds}-Fold Cross-Validation",
        fold_iterator()
    )


def run_loso_cross_validation(args: Dict, X, Y, subjects, class_names: List[str]) -> ExperimentResults:
    """Leave-One-Subject-Out cross-validation"""
    unique_subjects = np.unique(subjects)
    
    def fold_iterator():
        for i, test_subject in enumerate(unique_subjects):
            yield test_subject, test_subject
    
    return run_experiment(
        args, X, Y, subjects, class_names,
        prepare_data_loaders_loso,
        "LOSO Cross-Validation",
        fold_iterator()
    )


def get_model_config(model_name: str) -> Dict:
    """Get model-specific configuration"""
    configs = {
        'eegnet': {
            'time_steps': 250, 'num_electrodes': 64, 'F1': 16, 'F2': 32, 'D': 4,
            'kernel_1': 64, 'kernel_2': 16, 'dropout': 0.15, 'momentum': 0.01,
            'learning_rate': 0.001
        },
        'niceeeg': {
            'time_steps': 250, 'num_electrodes': 64, 'k': 40, 'm1': 25, 'm2': 51,
            's': 5, 'proj_dim': 768, 'dropout': 0.5, 'learning_rate': 0.001
        },
        'nettrast': {
            'time_steps': 250, 'num_electrodes': 64, 'embed_dim': 64, 'kernel_num': 64,
            'kernel_size': 3, 'nhead': 8, 'dim_feedforward': 256, 'num_layers': 3,
            'vocab_size': 128, 'dropout': 0.5, 'learning_rate': 0.001
        }
    }
    return configs.get(model_name, {})


def main(data_type: str = "audio"):
    """Main execution function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data setup
    if data_type == "audio":
        subjects = ['s2', 's3', 's4', 's5', 's8', 's10', 's12', 's7', 's9', 's11', 's6']
        edf_paths = [f"{sub}_audio{i} Data.edf" for sub in subjects for i in [1, 2]]
    else:
        subjects = ['s1', 's2', 's3', 's4', 's5', 's8', 's10', 's12', 's7', 's9', 's11', 's6']
        edf_paths = [f"{sub}_read{i} Data.edf" for sub in subjects for i in [1, 2]]
    
    edf_paths = [f for f in edf_paths if os.path.exists(f)]
    print(f"Found {len(edf_paths)} EDF files")
    
    # Load data
    try:
        X, Y, subject_ids, class_names, _ = load_and_preprocess_data_with_subjects(
            edf_paths, session_type=data_type
        )
        print(f"Data loaded - X: {X.shape}, Y: {np.bincount(Y)}, Subjects: {np.unique(subject_ids)}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Model setup
    model_name = 'eegnet'
    model_config = get_model_config(model_name)
    
    args = {
        'model_name': model_name,
        'class_num': len(class_names),
        'batch_size': 32,
        'epochs': 100,
        'lr': model_config.get('learning_rate', 0.001),
        'save_dir': f'{model_name}_experiments',
        'device': device,
        **model_config
    }
    
    print(f"\n{'='*60}")
    print(f"Running experiments with {model_name.upper()}")
    print(f"{'='*60}")
    
    # Run all experiments
    run_loso_cross_validation(args, X, Y, subject_ids, class_names)
    run_cross_validation(args, X, Y, subject_ids, class_names, n_folds=5)
    run_within_subject_evaluation(args, X, Y, subject_ids, class_names)


if __name__ == "__main__":
    main(data_type="audio") 