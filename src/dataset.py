import torch
from torch.utils.data import Dataset
class PfamDataset(Dataset):
    """Dataset object for loading the protein sequences

    Args:
        Dataset (Torch Dataset): _description_
    """
    def __init__(self, sequences, names,y):
        self.sequences = sequences
        self.targets = torch.LongTensor(y)
        self.names = names
    def __getitem__(self, index):
        return self.names[index], self.sequences[index], self.targets[index]
    def __len__(self):
        return len(self.names)
