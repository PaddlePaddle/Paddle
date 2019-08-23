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
from paddle.check_import_scipy import check_import_scipy

check_import_scipy(os.name)

try:
    from paddle.version import full_version as __version__
    from paddle.version import commit as __git_commit__

except ImportError:
    import sys
    sys.stderr.write('''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
                     )

import paddle.reader
import paddle.dataset
import paddle.batch
import paddle.compat
import paddle.distributed
batch = batch.batch
