#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import shutil
import glob


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    if os.path.isdir(src):  #copy directory
        pathList = os.path.split(src)
        dst = os.path.join(dst, pathList[-1])
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print("first copy directory: {0} --->>> {1}".format(src, dst))
        else:
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print("overwritten copy directory: {0} --->>> {1}".format(src, dst))
    else:  #copy file, wildcard
        if not os.path.exists(dst):
            os.makedirs(dst)
        srcFiles = glob.glob(src)
        for srcFile in srcFiles:
            shutil.copy(srcFile, dst)
            print("copy file: {0} --->>> {1}".format(srcFile, dst))


if __name__ == "__main__":
    main()
