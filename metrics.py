import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch


class Metrics:
    def __init__(self, column_names):
        column_names.insert(0, "time_stamp")
        self.df = pd.DataFrame(columns=column_names)

    def add_row(self, row_list):
        row_list.insert(0, str(datetime.datetime.now()))
        self.df.loc[len(self.df)] = row_list

    def save_to_csv(self, filepath):
        self.df.to_csv(filepath, index=False)
        
def evaluation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy 

def get_confusion_matrix(model, data_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds) 

    class_accs = []
    for i in range(len(class_names)):
        tf = cm[i, i]
        total = np.sum(cm[i, :])
        acc = tf / total if total > 0 else 0
        class_accs.append(acc)
    return cm, class_accs 

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap ="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def track_best_test_acc(model, test_loader, criterion, device, best_test_acc, best_test_epoch, epoch):
    test_loss, test_acc = evaluation(model, test_loader, criterion, device)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_epoch = epoch
    return test_loss, test_acc, best_test_acc, best_test_epoch 