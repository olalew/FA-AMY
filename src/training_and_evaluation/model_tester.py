import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.bioinformatics_dataset import BioinformaticsDataset
from src.model.fa_amy_module import FAAmyModule
from src.tools.metrics_scorer import MetricsScorer


class ModelTester:

    def __init__(self, pos_data_path: str, neg_data_path: str):
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, model_path: str):
        neg_test = np.load(self.neg_data_path)
        pos_test = np.load(self.pos_data_path)
        df_prot = np.concatenate((pos_test, neg_test), axis=0)
        ones = np.ones(len(pos_test), dtype=int)
        zeros = np.zeros(len(neg_test), dtype=int)
        df_lb = np.concatenate((ones, zeros))

        # Dataset and DataLoader Initialization
        test_set = BioinformaticsDataset(df_lb, df_prot)
        test_load = DataLoader(dataset=test_set, batch_size=8, shuffle=False)

        # Model Initialization
        model = FAAmyModule()
        model = model.to(self.device)

        print("========================== Test RESULT ================================")
        final_model_path = os.path.join(model_path, f"final_best_model.pth")
        model.load_state_dict(
            torch.load(final_model_path, map_location=self.device))
        model.eval()

        arr_labels = []
        arr_labels_hyps = []

        with torch.no_grad():
            for prot_x, data_y, length in test_load:
                prot_x, data_y, length = prot_x.to(self.device), data_y.to(self.device), length.to(self.device)
                y_pred = torch.sigmoid(model(prot_x))
                arr_labels.extend(data_y.cpu().numpy())
                arr_labels_hyps.extend(y_pred.cpu().numpy())

        labels = np.array(arr_labels)
        probs = np.array(arr_labels_hyps)

        scores = MetricsScorer.calculate_metrics(labels, probs)
        scores.print_metrics()