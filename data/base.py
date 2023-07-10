import torch.utils.data as data



class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return self.__class__.__name__

    def __len__(self):
        raise NotImplementedError()
