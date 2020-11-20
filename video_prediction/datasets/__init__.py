from .base_dataset import BaseVideoDataset
from .earthnet import EarthNet

def get_dataset_class(dataset):
    dataset_mappings = {
        'earthnet': 'EarthNet'
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    if dataset_class is None or not issubclass(dataset_class, BaseVideoDataset):
        raise ValueError('Invalid dataset %s' % dataset)
    return dataset_class
