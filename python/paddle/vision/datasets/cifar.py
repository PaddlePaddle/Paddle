#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import pickle
import tarfile

import numpy as np
from PIL import Image

import paddle
from paddle.dataset.common import _check_exists_and_download
from paddle.io import Dataset
=======
from __future__ import print_function

import tarfile
import numpy as np
import six
from PIL import Image
from six.moves import cPickle as pickle

import paddle
from paddle.io import Dataset
from paddle.dataset.common import _check_exists_and_download
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

MODE_FLAG_MAP = {
    'train10': 'data_batch',
    'test10': 'test_batch',
    'train100': 'train',
    'test100': 'test',
}


class Cifar10(Dataset):
    """
    Implementation of `Cifar-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 10 categories.

    Args:
        data_file (str, optional): Path to data file, can be set None if
            :attr:`download` is True. Default None, default data path: ~/.cache/paddle/dataset/cifar
        mode (str, optional): Either train or test mode. Default 'train'.
        transform (Callable, optional): transform to perform on image, None for no transform. Default: None.
        download (bool, optional): download dataset automatically if :attr:`data_file` is None. Default True.
        backend (str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}.
            If this option is not set, will get backend from :ref:`paddle.vision.get_image_backend <api_vision_image_get_image_backend>`,
            default backend is 'pil'. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of Cifar10 dataset.

    Examples:

        .. code-block:: python

            import itertools
            import paddle.vision.transforms as T
            from paddle.vision.datasets import Cifar10


            cifar10 = Cifar10()
            print(len(cifar10))
            # 50000

            for i in range(5):  # only show first 5 images
                img, label = cifar10[i]
                # do something with img and label
                print(type(img), img.size, label)
                # <class 'PIL.Image.Image'> (32, 32) 6


            transform = T.Compose(
                [
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        to_rgb=True,
                    ),
                ]
            )

            cifar10_test = Cifar10(
                mode="test",
                transform=transform,  # apply transform to every image
                backend="cv2",  # use OpenCV as image transform backend
            )
            print(len(cifar10_test))
            # 10000

            for img, label in itertools.islice(iter(cifar10_test), 5):  # only show first 5 images
                # do something with img and label
                print(type(img), img.shape, label)
                # <class 'paddle.Tensor'> [3, 64, 64] 3
    """

    def __init__(
        self,
        data_file=None,
        mode='train',
        transform=None,
        download=True,
        backend=None,
    ):
        assert mode.lower() in [
            'train',
            'test',
        ], "mode.lower() should be 'train' or 'test', but got {}".format(mode)
        self.mode = mode.lower()

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}".format(
                    backend
                )
            )
        self.backend = backend

        self._init_url_md5_flag()

        self.data_file = data_file
        if self.data_file is None:
            assert (
                download
            ), "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, self.data_url, self.data_md5, 'cifar', download
            )

        self.transform = transform

        # read dataset into memory
        self._load_data()

        self.dtype = paddle.get_default_dtype()

    def _init_url_md5_flag(self):
        self.data_url = CIFAR10_URL
        self.data_md5 = CIFAR10_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '10']

    def _load_data(self):
        self.data = []
        with tarfile.open(self.data_file, mode='r') as f:
            names = (
                each_item.name for each_item in f if self.flag in each_item.name
            )

            names = sorted(list(names))

            for name in names:
                batch = pickle.load(f.extractfile(name), encoding='bytes')

<<<<<<< HEAD
                data = batch[b'data']
                labels = batch.get(b'labels', batch.get(b'fine_labels', None))
                assert labels is not None
                for sample, label in zip(data, labels):
=======
                data = batch[six.b('data')]
                labels = batch.get(
                    six.b('labels'), batch.get(six.b('fine_labels'), None)
                )
                assert labels is not None
                for sample, label in six.moves.zip(data, labels):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    self.data.append((sample, label))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, np.array(label).astype('int64')

        return image.astype(self.dtype), np.array(label).astype('int64')

    def __len__(self):
        return len(self.data)


class Cifar100(Cifar10):
    """
    Implementation of `Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 100 categories.

    Args:
        data_file (str, optional): path to data file, can be set None if
            :attr:`download` is True. Default: None, default data path: ~/.cache/paddle/dataset/cifar
        mode (str, optional): Either train or test mode. Default 'train'.
        transform (Callable, optional): transform to perform on image, None for no transform. Default: None.
        download (bool, optional): download dataset automatically if :attr:`data_file` is None. Default True.
        backend (str, optional): Specifies which type of image to be returned:
            PIL.Image or numpy.ndarray. Should be one of {'pil', 'cv2'}.
            If this option is not set, will get backend from :ref:`paddle.vision.get_image_backend <api_vision_image_get_image_backend>`,
            default backend is 'pil'. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of Cifar100 dataset.

    Examples:

        .. code-block:: python

            import itertools
            import paddle.vision.transforms as T
            from paddle.vision.datasets import Cifar100


            cifar100 = Cifar100()
            print(len(cifar100))
            # 50000

            for i in range(5):  # only show first 5 images
                img, label = cifar100[i]
                # do something with img and label
                print(type(img), img.size, label)
                # <class 'PIL.Image.Image'> (32, 32) 19


            transform = T.Compose(
                [
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        to_rgb=True,
                    ),
                ]
            )

            cifar100_test = Cifar100(
                mode="test",
                transform=transform,  # apply transform to every image
                backend="cv2",  # use OpenCV as image transform backend
            )
            print(len(cifar100_test))
            # 10000

            for img, label in itertools.islice(iter(cifar100_test), 5):  # only show first 5 images
                # do something with img and label
                print(type(img), img.shape, label)
                # <class 'paddle.Tensor'> [3, 64, 64] 49
    """

    def __init__(
        self,
        data_file=None,
        mode='train',
        transform=None,
        download=True,
        backend=None,
    ):
<<<<<<< HEAD
        super().__init__(data_file, mode, transform, download, backend)
=======
        super(Cifar100, self).__init__(
            data_file, mode, transform, download, backend
        )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _init_url_md5_flag(self):
        self.data_url = CIFAR100_URL
        self.data_md5 = CIFAR100_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '100']
