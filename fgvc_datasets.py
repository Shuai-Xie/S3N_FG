import os
from collections import OrderedDict
from typing import Callable, Optional
from torch.utils.data import Dataset
from PIL import Image


class FGVC_Dataset(Dataset):
    """
    Fine-grained visual classification dataset.
    """

    def __init__(self, data_dir, split=None, label_path=None, transform=None, target_transform=None):
        """
        :param data_dir:
        :param split:
        :param label_path:
        :param transform:
        :param target_transform:
        """
        self.data_dir = data_dir
        self.split = split
        self.label_path = label_path
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(self.data_dir, 'images')
        self.image_labels = self._read_annotation(self.split)

    def _read_annotation(self, split):
        class_labels = OrderedDict()
        if self.label_path is None:
            label_path = os.path.join(self.data_dir, split + '.txt')
        else:
            label_path = os.path.join(self.data_dir, self.label_path, split + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    name, label = line.split(' ')
                    class_labels[name] = int(label)
        else:
            raise NotImplementedError(
                'Invalid path for dataset')

        return list(class_labels.items())

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_labels)


def fgvc_dataset(
        data_dir: str,
        split: str,
        label_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None) -> Dataset:
    '''Fine-grained visual classification datasets.
    '''
    return FGVC_Dataset(data_dir, split, label_path, transform, target_transform)
