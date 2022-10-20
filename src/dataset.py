from torch.utils.data import Dataset
import torch

class PfamDataset(Dataset):
    def __init__(self, sequences, names,y, transform=None):
        self.sequences = sequences
        self.targets = torch.LongTensor(y)
        self.names = names
    def __getitem__(self, index):
        return self.names[index], self.sequences[index], self.targets[index]
    def __len__(self):
        return len(self.names)


