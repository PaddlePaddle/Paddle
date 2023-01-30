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

<<<<<<< HEAD
import glob
import os
import shutil
import sys
=======
import os
import sys
import shutil
import glob
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
<<<<<<< HEAD
    if os.path.isdir(src):  # copy directory
=======
    if os.path.isdir(src):  #copy directory
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        pathList = os.path.split(src)
        dst = os.path.join(dst, pathList[-1])
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print("first copy directory: {0} --->>> {1}".format(src, dst))
        else:
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print("overwritten copy directory: {0} --->>> {1}".format(src, dst))
<<<<<<< HEAD
    else:  # copy file, wildcard
=======
    else:  #copy file, wildcard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not os.path.exists(dst):
            os.makedirs(dst)
        srcFiles = glob.glob(src)
        for srcFile in srcFiles:
            shutil.copy(srcFile, dst)
            print("copy file: {0} --->>> {1}".format(srcFile, dst))


if __name__ == "__main__":
    main()
