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

__all__ = models.__all__ \
        + transforms.__all__ \
        + datasets.__all__

_image_backend = 'pil'


def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (str): Name of the image backend. one of {'pil', 'cv2'}.
    """
    global _image_backend
    if backend not in ['pil', 'cv2']:
        raise ValueError("Invalid backend '{}'. Options are 'pil' and 'cv2'"
                         .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend
