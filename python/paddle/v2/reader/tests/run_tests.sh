#!/bin/bash
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


pushd `dirname $0` > /dev/null
SCRIPTPATH=$PWD
popd > /dev/null

cd $SCRIPTPATH
$1 -m pip install ../../../../../paddle/dist/*.whl

test_list="creator_test.py decorator_test.py"

export PYTHONPATH=$PWD/../../../../../python/

for fn in $test_list
do
  echo "test $fn"
  $1 $fn
  if [ $? -ne 0 ]; then
    exit 1
  fi
done
