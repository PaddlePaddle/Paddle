# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import six.moves.cPickle as pickle
import random
import collections


def save_file(data, filename):
    """
    Save data into pickle format.
    data: the data to save.
    filename: the output filename.
    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def save_list(l, outfile):
    """
    Save a list of string into a text file. There is one line for each string.
    l: the list of string to save
    outfile: the output file
    """
    open(outfile, "w").write("\n".join(l))


def exclude_pattern(f):
    """
    Return whether f is in the exclude pattern.
    Exclude the files that starts with . or ends with ~.
    """
    return f.startswith(".") or f.endswith("~")


def list_dirs(path):
    """
    Return a list of directories in path. Exclude all the directories that
    start with '.'.
    path: the base directory to search over.
    """
    return [
        os.path.join(path, d) for d in next(os.walk(path))[1]
        if not exclude_pattern(d)
    ]


def list_images(path, exts=set(["jpg", "png", "bmp", "jpeg"])):
    """
    Return a list of images in path.
    path: the base directory to search over.
    exts: the extensions of the images to find.
    """
    return [os.path.join(path, d) for d in  os.listdir(path) \
            if os.path.isfile(os.path.join(path, d)) and not exclude_pattern(d)\
            and os.path.splitext(d)[-1][1:] in exts]


def list_files(path):
    """
    Return a list of files in path.
    path: the base directory to search over.
    exts: the extensions of the images to find.
    """
    return [os.path.join(path, d) for d in  os.listdir(path) \
            if os.path.isfile(os.path.join(path, d)) and not exclude_pattern(d)]


def get_label_set_from_dir(path):
    """
    Return a dictionary of the labels and label ids from a path.
    Assume each directory in the path corresponds to a unique label.
    The keys of the dictionary is the label name.
    The values of the dictionary is the label id.
    """
    dirs = list_dirs(path)
    return dict([(os.path.basename(d), i) for i, d in enumerate(sorted(dirs))])


class Label:
    """
    A class of label data.
    """

    def __init__(self, label, name):
        """
        label: the id of the label.
        name: the name of the label.
        """
        self.label = label
        self.name = name

    def convert_to_paddle_format(self):
        """
        convert the image into the paddle batch format.
        """
        return int(self.label)

    def __hash__(self):
        return hash((self.label))


class Dataset:
    """
    A class to represent a dataset. A dataset contains a set of items.
    Each item contains multiple slots of data.
    For example: in image classification dataset, each item contains two slot,
    The first slot is an image, and the second slot is a label.
    """

    def __init__(self, data, keys):
        """
        data: a list of data.
              Each data is a tuple containing multiple slots of data.
              Each slot is an object with convert_to_paddle_format function.
        keys: contains a list of keys for all the slots.
        """
        self.data = data
        self.keys = keys

    def check_valid(self):
        for d in self.data:
            assert (len(d) == len(self.keys))

    def permute(self, key_id, num_per_batch):
        """
        Permuate data for batching. It supports two types now:
        1. if key_id == None, the batching process is completely random.
        2. if key_id is not None. The batching process Permuate the data so that the key specified by key_id are
        uniformly distributed in batches. See the comments of permute_by_key for details.
        """
        if key_id is None:
            self.uniform_permute()
        else:
            self.permute_by_key(key_id, num_per_batch)

    def uniform_permute(self):
        """
        Permuate the data randomly.
        """
        random.shuffle(self.data)

    def permute_by_key(self, key_id, num_per_batch):
        """
        Permuate the data so that the key specified by key_id are
        uniformly distributed in batches.
        For example: if we have three labels, and the number of data
        for each label are 100, 200, and 300, respectively.  The number of batches is 4.
        Then, the number of data for these labels is 25, 50, and 75.
        """
        # Store the indices of the data that has the key value
        # specified by key_id.
        keyvalue_indices = collections.defaultdict(list)
        for idx in range(len(self.data)):
            keyvalue_indices[self.data[idx][key_id].label].append(idx)
        for k in keyvalue_indices:
            random.shuffle(keyvalue_indices[k])

        num_data_per_key_batch = \
            math.ceil(num_per_batch / float(len(list(keyvalue_indices.keys()))))

        if num_data_per_key_batch < 2:
            raise Exception("The number of data in a batch is too small")

        permuted_data = []
        keyvalue_readpointer = collections.defaultdict(int)
        while len(permuted_data) < len(self.data):
            for k in keyvalue_indices:
                begin_idx = keyvalue_readpointer[k]
                end_idx = int(
                    min(begin_idx + num_data_per_key_batch,
                        len(keyvalue_indices[k])))
                print("begin_idx, end_idx")
                print(begin_idx, end_idx)
                for idx in range(begin_idx, end_idx):
                    permuted_data.append(self.data[keyvalue_indices[k][idx]])
                keyvalue_readpointer[k] = end_idx
        self.data = permuted_data


class DataBatcher:
    """
    A class that is used to create batches for both training and testing
    datasets.
    """

    def __init__(self, train_data, test_data, label_set):
        """
        train_data, test_data: Each one is a dataset object representing
        training and testing data, respectively.
        label_set: a dictionary storing the mapping from label name to label id.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.label_set = label_set
        self.num_per_batch = 5000
        assert (self.train_data.keys == self.test_data.keys)

    def create_batches_and_list(self, output_path, train_list_name,
                                test_list_name, label_set_name):
        """
        Create batches for both training and testing objects.
        It also create train.list and test.list to indicate the list
        of the batch files for training and testing data, respectively.
        """
        train_list = self.create_batches(self.train_data, output_path, "train_",
                                         self.num_per_batch)
        test_list = self.create_batches(self.test_data, output_path, "test_",
                                        self.num_per_batch)
        save_list(train_list, os.path.join(output_path, train_list_name))
        save_list(test_list, os.path.join(output_path, test_list_name))
        save_file(self.label_set, os.path.join(output_path, label_set_name))

    def create_batches(self,
                       data,
                       output_path,
                       prefix="",
                       num_data_per_batch=5000):
        """
        Create batches for a Dataset object.
        data: the Dataset object to process.
        output_path: the output path of the batches.
        prefix: the prefix of each batch.
        num_data_per_batch: number of data in each batch.
        """
        num_batches = int(math.ceil(len(data.data) / float(num_data_per_batch)))
        batch_names = []
        data.check_valid()
        num_slots = len(data.keys)
        for i in range(num_batches):
            batch_name = os.path.join(output_path, prefix + "batch_%03d" % i)
            out_data = dict([(k, []) for k in data.keys])
            begin_idx = i * num_data_per_batch
            end_idx = min((i + 1) * num_data_per_batch, len(data.data))
            for j in range(begin_idx, end_idx):
                for slot_id in range(num_slots):
                    out_data[data.keys[slot_id]].\
                        append(data.data[j][slot_id].convert_to_paddle_format())
            save_file(out_data, batch_name)
            batch_names.append(batch_name)
        return batch_names


class DatasetCreater(object):
    """
    A virtual class for creating datasets.
    The derived class needs to implement the following methods:
       - create_dataset()
       - create_meta_file()
    """

    def __init__(self, data_path):
        """
        data_path: the path to store the training data and batches.
        train_dir_name: relative training data directory.
        test_dir_name: relative testing data directory.
        batch_dir_name: relative batch directory.
        num_per_batch: the number of data in a batch.
        meta_filename: the filename of the meta file.
        train_list_name: training batch list name.
        test_list_name: testing batch list name.
        label_set: label set name.
        overwrite: whether to overwrite the files if the batches are already in
                   the given path.
        """
        self.data_path = data_path
        self.train_dir_name = 'train'
        self.test_dir_name = 'test'
        self.batch_dir_name = 'batches'
        self.num_per_batch = 50000
        self.meta_filename = "batches.meta"
        self.train_list_name = "train.list"
        self.test_list_name = "test.list"
        self.label_set_name = "labels.pkl"
        self.output_path = os.path.join(self.data_path, self.batch_dir_name)
        self.overwrite = False
        self.permutate_key = "labels"
        self.from_list = False

    def create_meta_file(self, data):
        """
        Create a meta file from training data.
        data: training data given in a Dataset format.
        """
        raise NotImplementedError

    def create_dataset(self, path):
        """
        Create a data set object from a path.
        It will use directory structure or a file list to determine dataset if
        self.from_list is True. Otherwise, it will uses a file list  to
        determine the dataset.
        path: the path of the dataset.
        return a tuple of Dataset object, and a mapping from label set
        to label id.
        """
        if self.from_list:
            return self.create_dataset_from_list(path)
        else:
            return self.create_dataset_from_dir(path)

    def create_dataset_from_list(self, path):
        """
        Create a data set object from a path.
        It will uses a file list to determine the dataset.
        path: the path of the dataset.
        return a tuple of Dataset object, and a mapping from label set
        to label id
        """
        raise NotImplementedError

    def create_dataset_from_dir(self, path):
        """
        Create a data set object from a path.
        It will use directory structure or a file list to determine dataset if
        self.from_list is True.
        path: the path of the dataset.
        return a tuple of Dataset object, and a mapping from label set
        to label id
        """
        raise NotImplementedError

    def create_batches(self):
        """
        create batches and meta file.
        """
        train_path = os.path.join(self.data_path, self.train_dir_name)
        test_path = os.path.join(self.data_path, self.test_dir_name)
        out_path = os.path.join(self.data_path, self.batch_dir_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if (self.overwrite or not os.path.exists(
                os.path.join(out_path, self.train_list_name))):
            train_data, train_label_set = \
                self.create_dataset(train_path)
            test_data, test_label_set = \
                self.create_dataset(test_path)

            train_data.permute(
                self.keys.index(self.permutate_key), self.num_per_batch)

            assert (train_label_set == test_label_set)
            data_batcher = DataBatcher(train_data, test_data, train_label_set)
            data_batcher.num_per_batch = self.num_per_batch
            data_batcher.create_batches_and_list(
                self.output_path, self.train_list_name, self.test_list_name,
                self.label_set_name)
            self.num_classes = len(list(train_label_set.keys()))
            self.create_meta_file(train_data)
        return out_path
