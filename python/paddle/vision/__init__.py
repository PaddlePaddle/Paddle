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

from . import models
from .models import *

from . import transforms
from .transforms import *

from . import datasets
from .datasets import *

__all__ = ['set_image_backend', 'get_image_backend']

__all__ += models.__all__ \
        + transforms.__all__ \
        + datasets.__all__

_image_backend = 'pil'


def set_image_backend(backend):
    """
    Specifies the backend used to load images in class ``paddle.vision.datasets.ImageFolder`` 
    and ``paddle.vision.datasets.DatasetFolder`` . Now support backends are pillow and opencv. 
    If backend not set, will use 'pil' as default. 

    Args:
        backend (str): Name of the image load backend, should be one of {'pil', 'cv2'}.

    Examples:
    
        .. code-block:: python

            import os
            import cv2
            import shutil
            import tempfile
            import numpy as np

            from paddle.vision import DatasetFolder
            from paddle.vision import set_image_backend

            set_image_backend('cv2')

            def make_fake_dir():
                data_dir = tempfile.mkdtemp()

                for i in range(2):
                    sub_dir = os.path.join(data_dir, 'class_' + str(i))
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir)
                    for j in range(2):
                        fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                        cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)
                return data_dir

            temp_dir = make_fake_dir()

            cv2_data_folder = DatasetFolder(temp_dir)

            for items in cv2_data_folder:
                break

            # should get numpy.ndarray
            print(type(items[0]))

            set_image_backend('pil')

            pil_data_folder = DatasetFolder(temp_dir)

            for items in pil_data_folder:
                break

            # should get PIL.Image.Image
            print(type(items[0]))

            shutil.rmtree(temp_dir)
    """
    global _image_backend
    if backend not in ['pil', 'cv2']:
        raise ValueError(
            "Expected backend are one of {'pil', 'cv2'}, but got {}"
            .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images

    Returns:
        str: backend of image load.

    Examples:
    
        .. code-block:: python

            from paddle.vision import get_image_backend

            backend = get_image_backend()
            print(backend)

    """
    return _image_backend
