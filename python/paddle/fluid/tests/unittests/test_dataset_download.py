# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
<<<<<<< HEAD

from paddle.dataset.common import DATA_HOME, download, md5file


class TestDataSetDownload(unittest.TestCase):
=======
from paddle.dataset.common import download, DATA_HOME, md5file


class TestDataSetDownload(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        flower_path = DATA_HOME + "/flowers/imagelabels.mat"

        if os.path.exists(flower_path):
            os.remove(flower_path)

    def test_download_url(self):
        LABEL_URL = 'http://paddlemodels.bj.bcebos.com/flowers/imagelabels.mat'
        LABEL_MD5 = 'e0620be6f572b9609742df49c70aed4d'

        catch_exp = False
        try:
            download(LABEL_URL, 'flowers', LABEL_MD5)
        except Exception as e:
            catch_exp = True

<<<<<<< HEAD
        self.assertTrue(not catch_exp)
=======
        self.assertTrue(catch_exp == False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        file_path = DATA_HOME + "/flowers/imagelabels.mat"

        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(md5file(file_path), LABEL_MD5)


if __name__ == '__main__':
    unittest.main()
