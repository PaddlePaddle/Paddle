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

USE_VIRTUALENV_FOR_TEST=$1; shift
PYTHON=$1; shift

if [ $USE_VIRTUALENV_FOR_TEST -ne 0 ]; then
   rm -rf .test_env
   virtualenv .test_env
   unset PYTHONHOME
   unset PYTHONPATH
   source .test_env/bin/activate
   PYTHON=python
fi

$PYTHON -m pip install $SCRIPTPATH/../dist/*.whl

if [ "X${PADDLE_PACKAGE_DIR}" != "X" ]; then
   $PYTHON -m pip install ${PADDLE_PACKAGE_DIR}/*.whl
else
   export PYTHONPATH=$SCRIPTPATH/../../python/
fi

$PYTHON -m pip install ipython==5.3

for fn in "$@"
do
  echo "test $fn"
  $PYTHON $fn
  if [ $? -ne 0 ]; then
    exit 1
  fi
done

if [ $USE_VIRTUALENV_FOR_TEST -ne 0 ]; then
    deactivate
    rm -rf .test_env
fi
