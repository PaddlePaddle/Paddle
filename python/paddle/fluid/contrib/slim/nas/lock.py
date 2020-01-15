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
__All__ = ['lock', 'unlock']
if os.name == 'nt':

    def lock(file):
        raise NotImplementedError('Windows is not supported.')

    def unlock(file):
        raise NotImplementedError('Windows is not supported.')

elif os.name == 'posix':
    from fcntl import flock, LOCK_EX, LOCK_UN

    def lock(file):
        """Lock the file in local file system."""
        flock(file.fileno(), LOCK_EX)

    def unlock(file):
        """Unlock the file in local file system."""
        flock(file.fileno(), LOCK_UN)
else:
    raise RuntimeError("File Locker only support NT and Posix platforms!")
