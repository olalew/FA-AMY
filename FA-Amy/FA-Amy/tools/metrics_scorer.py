import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


class EvaluationMetrics:
    AUC: float
    ACC: float
    BA: float
    SN: float
    SP: float
    MCC: float
    G_mean: float
    F1: float
    Pre: float
    TP: float
    TN: float
    FP: float
    FN: float

    def __init__(self, AUC: float, ACC: float, BA: float, SN: float, SP: float, MCC: float, G_mean: float,
                 F1: float, Pre: float, TP: float, TN: float, FP: float, FN: float):
        self.AUC = AUC
        self.ACC = ACC
        self.BA = BA
        self.SN = SN
        self.SP = SP
        self.MCC = MCC
        self.G_mean = G_mean
        self.F1 = F1
        self.Pre = Pre
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

    def print_metrics(self):
        print("####################### Model scores #######################")
        print(
            f"SN: {self.SN:.4f} | "
            f"SP: {self.SP:.4f} | "
            f"ACC: {self.ACC:.4f} | "
            f"BA: {self.BA:.4f} | "
            f"MCC: {self.MCC:.4f} | "
            f"Pre: {self.Pre:.4f} | "
            f"AUC: {self.AUC:.4f} | "
            f"G-mean: {self.G_mean:.4f} | "
            f"F1-score: {self.F1:.4f}"
        )

    def get_base_dictionary(self):
        return ["SN", "SP", "ACC", "BA", "MCC", "Pre", "Gmean", "F1", "AUROC"]

    def get_base_scores(self):
        return [self.SN, self.SP, self.ACC, self.BA, self.MCC, self.Pre, self.G_mean, self.F1, self.AUC]


class MetricsScorer:

    @staticmethod
    def calculate_metrics(labels, probs) -> EvaluationMetrics:
        AUC = roc_auc_score(labels, probs)
        pred = np.around(probs)
        TN, FP, FN, TP = confusion_matrix(labels, pred).ravel()

        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FN + FP)
        BA = (SN + SP) / 2
        MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
        Pre = TP / (TP + FP)
        G_mean = np.sqrt(SN * SP)
        F1 = 2 * Pre * SN / (Pre + SN)

        return EvaluationMetrics(
            AUC=AUC,
            ACC=ACC,
            BA=BA,
            SN=SN,
            SP=SP,
            MCC=MCC,
            G_mean=G_mean,
            F1=F1,
            Pre=Pre,
            TP=TP,
            TN=TN,
            FP=FP,
            FN=FN
        )
