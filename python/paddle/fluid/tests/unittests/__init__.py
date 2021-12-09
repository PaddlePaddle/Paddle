#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
# limitations under the License.p

# Note: On Windows, import form subdirectories such as dirA()->dirB(), current directory 
# will still be dirA(), But is should be dirB(). So it will ModulNotFoundError
# please refer to https://stackoverflow.com/questions/8953844/import-module-from-subfolder

import os
if os.name == 'nt':
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.insert(0, dirname)
    print(sys.path)
