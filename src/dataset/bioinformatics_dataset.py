import torch
from torch.utils.data import Dataset

class BioinformaticsDataset(Dataset):  #
    def __init__(self, label, prot, return_index=False):
        self.lb = label
        self.df_prot = prot
        self.return_index = return_index

    def __getitem__(self, index):
        prot = self.df_prot[index]
        prot = torch.tensor(prot, dtype=torch.float)
        label = self.lb[index]
        label = torch.tensor(label, dtype=torch.float)
        # The features extracted by ESM C have padding before and after
        data_length = 902
        if self.return_index:
            return prot, label, data_length, index
        return prot, label, data_length

    def __len__(self):
        return len(self.df_prot)