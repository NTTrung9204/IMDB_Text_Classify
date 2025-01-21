from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size(0)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    