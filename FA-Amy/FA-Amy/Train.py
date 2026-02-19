from torch.utils.data import Dataset, DataLoader, Subset
import os
import torch

from tools.metrics_scorer import MetricsScorer

torch.use_deterministic_algorithms(True, warn_only=False)
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy

from tools.random_seed import RandomSeed
from dataset.bioinformatics_dataset import BioinformaticsDataset
from model.fa_amy_module import FAAmyModule

RandomSeed.random_seed(777)


def train():
    device = torch.device("cuda")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    neg_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_neg_train.npy")
    pos_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_pos_train.npy")
    model_path = os.path.join(base_dir, "..", "Model-saved")

    neg_train = np.load(neg_data_path)
    pos_train = np.load(pos_data_path)

    df_prot = np.concatenate((pos_train, neg_train), axis=0)
    df_lb = np.concatenate((np.ones(len(pos_train), dtype=int), np.zeros(len(neg_train), dtype=int)))

    dataset = BioinformaticsDataset(df_lb, df_prot)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

    # Maximum of training epochs
    epochs = 100
    # Early Stopping patience
    patience = 10

    all_metrics = MetricsScorer.init_all_metrics()

    fold_best_models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_prot, df_lb)):
        print(f"\n--- Fold {fold + 1} ---")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)
        model = FAAmyModule().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        best_val_ba = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            for prot_x, data_y, length in train_loader:
                prot_x, data_y, length = prot_x.to(device), data_y.to(device), length.to(device)
                y_pred = model(prot_x)
                loss = F.binary_cross_entropy_with_logits(
                    y_pred.view(-1), data_y, pos_weight=torch.tensor([2.0]).to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            all_predictions, all_labels = [], []
            with torch.no_grad():
                for prot_x, data_y, length in val_loader:
                    prot_x, data_y, length = prot_x.to(device), data_y.to(device), length.to(device)
                    y_pred = torch.sigmoid(model(prot_x))
                    all_predictions.extend(y_pred.cpu().numpy())
                    all_labels.extend(data_y.cpu().numpy())

            labels = np.array(all_labels)
            probs = np.array(all_predictions)
            pred = np.around(probs)

            TN, FP, FN, TP = confusion_matrix(labels, pred).ravel()
            SN = TP / (TP + FN) if (TP + FN) != 0 else 0
            SP = TN / (TN + FP) if (TN + FP) != 0 else 0
            BA = (SN + SP) / 2

            # warm up
            min_epoch_to_save = 15
            # To select the model with the strongest discriminative ability, we used the balanced accuracy as the evaluation criterion.
            if BA > best_val_ba:
                if epoch >= min_epoch_to_save:
                    best_val_ba = BA
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"Epoch {epoch + 1}: Val BA improved to {BA:.4f}, model saved.")
                else:
                    print(f"Epoch {epoch + 1}: Val BA improved to {BA:.4f}, but still in warm-up (no save).")
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch + 1}: No improvement, wait = {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        best_model_path = os.path.join(model_path, f"best_fold{fold}.pth")
        torch.save(best_model_state, best_model_path)
        fold_best_models.append(best_model_path)

        model.load_state_dict(best_model_state)
        model.eval()
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for prot_x, data_y, length in val_loader:
                prot_x, data_y, length = prot_x.to(device), data_y.to(device), length.to(device)
                y_pred = torch.sigmoid(model(prot_x))
                all_predictions.extend(y_pred.cpu().numpy())
                all_labels.extend(data_y.cpu().numpy())

        labels = np.array(all_labels)
        probs = np.array(all_predictions)

        scores = MetricsScorer.calculate_metrics(labels, probs)
        scores.print_metrics()

        for metric, value in zip(scores.get_base_dictionary(), scores.get_base_scores()):
            all_metrics[metric].append(value)

    print("\n===== 5-Fold Cross Validation Summary =====")
    for metric in all_metrics:
        values = all_metrics[metric]
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    best_ba = -1
    best_model_path = None
    for i, path in enumerate(fold_best_models):
        ba = all_metrics["BA"][i]
        if ba > best_ba:
            best_ba = ba
            best_model_path = path

    print(f"\nBest model chosen from fold with BA = {best_ba:.4f}: {best_model_path}")
    final_model_path = os.path.join(model_path, f"final_best_model.pth")
    torch.save(torch.load(best_model_path), final_model_path)
    print(f"Final best model saved to {final_model_path}")


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    train()
