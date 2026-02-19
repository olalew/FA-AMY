import os
import torch

torch.use_deterministic_algorithms(True, warn_only=False)
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from tools.metrics_scorer import MetricsScorer
from tools.random_seed import RandomSeed

from dataset.bioinformatics_dataset import BioinformaticsDataset
from model.fa_amy_module import FAAmyModule

RandomSeed.random_seed(777)


def run_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_neg_test.npy")
    pos_data_path = os.path.join(base_dir, "..", "Dataset", "Benchmark_dataset", "new", "esmc_pos_test.npy")
    model_path = os.path.join(base_dir, "..", "Model-saved")
    neg_test = np.load(neg_data_path)
    pos_test = np.load(pos_data_path)
    df_prot = np.concatenate((pos_test, neg_test), axis=0)
    ones = np.ones(len(pos_test), dtype=int)
    zeros = np.zeros(len(neg_test), dtype=int)
    df_lb = np.concatenate((ones, zeros))

    # Dataset and DataLoader Initialization
    test_set = BioinformaticsDataset(df_lb, df_prot)
    test_load = DataLoader(dataset=test_set, batch_size=8, shuffle=False)

    # Model Initialization
    model = FAAmyModule()
    model = model.to(device)

    print("==========================Test RESULT================================")
    final_model_path = os.path.join(model_path, f"final_best_model.pth")
    model.load_state_dict(
        torch.load(final_model_path, map_location=device))
    model.eval()

    arr_labels = []
    arr_labels_hyps = []

    with torch.no_grad():
        for prot_x, data_y, length in test_load:
            prot_x, data_y, length = prot_x.to(device), data_y.to(device), length.to(device)
            y_pred = torch.sigmoid(model(prot_x))
            arr_labels.extend(data_y.cpu().numpy())
            arr_labels_hyps.extend(y_pred.cpu().numpy())

    labels = np.array(arr_labels)
    probs = np.array(arr_labels_hyps)

    scores = MetricsScorer.calculate_metrics(labels, probs)
    scores.print_metrics()


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    run_model()
