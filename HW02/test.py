import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LibriDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = None if y is None else y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


X = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([5, 6])
train_set = LibriDataset(X, y)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
for feature, label in train_loader:
    print(feature, label)
    print('------')
print('xxx')
