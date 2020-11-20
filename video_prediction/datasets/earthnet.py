import argparse
import glob
import itertools
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset

class EarthNet(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(EarthNet, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (128, 128, 12)
    def get_default_hparams_dict(self):
        default_hparams = super(EarthNet, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,
            sequence_length=30,
            force_time_shift=True,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        return len(self.filenames)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
