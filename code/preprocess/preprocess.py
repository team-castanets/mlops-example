import argparse
import os
from tqdm import tqdm
from glob import glob
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import tfds_korean.klue_sts

from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath


# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--output-path', type=str, default='klue/klue.tfrecord', help='Output path')
# fmt: on


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(sentence1, sentence2, label):
    feature = {
        'sentence1': _bytes_feature(sentence1),
        'sentence2': _bytes_feature(sentence2),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    # Load KLUE STS dataset
    dataset = tfds.load("klue_sts")
    with tf.io.TFRecordWriter(args.output_path) as writer:
        for data in tqdm(dataset['train']):
            tf_example = make_example(
                data['sentence1'].numpy(),
                data['sentence2'].numpy(),
                data['binary-label'].numpy()
            )
            writer.write(tf_example.SerializeToString())

    # Save as TFRecord dataset
    ws = Workspace.from_config()
    datastore = Datastore.get(ws, 'datasets')
    ds = Dataset.File.upload_directory(
        src_dir='klue',
        target=DataPath(datastore, 'klue'),
        show_progress=True,
        overwrite=True,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
