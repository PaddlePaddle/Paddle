# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def _get_user_home():
    return os.path.expanduser('~')


def _get_paddleaudio_home():
    if 'PPAUDIO_DATA' in os.environ:
        home_path = os.environ['PPAUDIO_DATA']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPAUDIO_HOME {} is not a directory.'
                    .format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddle_audiodata')


def _get_sub_home(directory):
    home = os.path.join(_get_paddleaudio_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=true)
    return home


PPAUDIO_HOME = _get_paddleaudio_home()
DATA_HOME = _get_sub_home('datasets')
