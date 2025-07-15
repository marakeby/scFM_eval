
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        H = self.feature_extractor(x)
        A = self.attention(H)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        out = self.classifier(M)
        return out, A

class MILExperiment:
    def __init__(self, embedding_col, label_key="outcome", patient_key="patient_id", celltype_key="cell_type", hidden_dim=128, epochs=40, lr=1e-3):
        # self.adata = adata
        self.embedding_col = embedding_col
        self.label_key = label_key
        self.patient_key = patient_key
        self.celltype_key = celltype_key
        self.model = None
        
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr

    # def split_data(self, test_size=0.2, random_state=42):
    #     patient_ids = self.adata.obs[self.patient_key].unique()
    #     train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=random_state)
    #     self.adata_train = self.adata[self.adata.obs[self.patient_key].isin(train_ids)].copy()
    #     self.adata_test = self.adata[self.adata.obs[self.patient_key].isin(test_ids)].copy()

    def prepare_bags(self, adata):
        self.input_dim = adata.obsm[self.embedding_col].shape[1]
        bags, labels, meta = [], [], []
        pids=[]
        for pid in adata.obs[self.patient_key].unique():
            adata_p = adata[adata.obs[self.patient_key] == pid]
            # X = adata_p[:, self.feature_keys].X
            X= adata_p.obsm[self.embedding_col]
            if hasattr(X, 'toarray'):
                X = X.toarray()
            y = adata_p.obs[self.label_key].iloc[0]
            bags.append(torch.tensor(X, dtype=torch.float32))
            labels.append(torch.tensor(y, dtype=torch.float32))
            # meta.append(adata_p.obs[self.celltype_key].values)
            meta = None
            pids.append(pid)
        return pids, bags, labels, meta

    def train(self, adata_train ):
        pids, bags_train, labels_train, _ = self.prepare_bags(adata_train)
        input_dim = self.input_dim
        # self.model = AttentionMIL(input_dim=len(self.feature_keys), hidden_dim=self.hidden_dim)
        self.model = AttentionMIL(input_dim=input_dim, hidden_dim=self.hidden_dim)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in zip(bags_train, labels_train):
                self.model.train()
                optimizer.zero_grad()
                out, _ = self.model(x)
                loss = criterion(out.view(-1), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(bags_train):.4f}")

    def evaluate(self, adata_test):
        pids, bags_test, labels_test, _ = self.prepare_bags(adata_test)
        self.model.eval()
        pred_scores = []
        with torch.no_grad():
            for x in bags_test:
                out, _ = self.model(x)
                pred_scores.append(torch.sigmoid(out).item())
        y_true = [l.item() for l in labels_test]
        # if len(np.unique(y_true)) < 2:
        #     auc= None
        # else:
        #     auc = roc_auc_score(y_true, preds)
        # print("Test AUC:", auc)
        pred_scores = np.array(pred_scores)
        preds = pred_scores>=0.5
        return pids, y_true, preds, pred_scores

    def visualize_attention(self):
        bags_test, _, metas = self.prepare_bags(self.adata_test)
        attention_weights_by_celltype = []
        self.model.eval()
        with torch.no_grad():
            for x, meta in zip(bags_test, metas):
                _, A = self.model(x)
                A = A.squeeze().numpy()
                df = pd.DataFrame({'attention': A, 'cell_type': meta})
                attention_weights_by_celltype.append(df)
        combined_df = pd.concat(attention_weights_by_celltype)
        summary = combined_df.groupby('cell_type')['attention'].mean().sort_values(ascending=False)
        summary.plot(kind='bar', title='Average Attention by Cell Type')
        plt.ylabel("Average Attention Weight")
        plt.tight_layout()
        plt.show()
        return summary

    def plot_evaluation_curves(self, y_true, y_pred, threshold=0.5, plot_prefix='mil_plot'):
        # Confusion Matrix
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]

        cm = confusion_matrix(y_true, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots();
        disp.plot(ax=ax)
        plot_title = 'confusion_matrix';
        plt.title("Confusion Matrix")
        plt.savefig(f'{plot_prefix}_{plot_title}.png');
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plot_title = 'roc_curve';
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f'{plot_prefix}_{plot_title}.png');
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plot_title = 'pr_curve';
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid()
        plt.savefig(f'{plot_prefix}_{plot_title}.png');
        plt.close()
