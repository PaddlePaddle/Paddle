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
echo python: $PYTHON
pip list

if [ $USE_VIRTUALENV_FOR_TEST -ne 0 ]; then
   rm -rf .test_env
   virtualenv .test_env
   source .test_env/bin/activate
   PYTHON=python
fi

$PYTHON -m pip install $SCRIPTPATH/../dist/*.whl
# The next line is for debug, will be deleted
$PYTHON -m pip list
$PYTHON -m pip install requests matplotlib numpy ipython==5.3
$PYTHON -m pip list
echo $PYTHON
echo PYTHONPATH: $PYTHONPATH
ls $PYTHONPATH
python -c 'import numpy; import pkgutil; print numpy.__path__; print(str(list(pkgutil.iter_modules(numpy.__path__))))'
python -c 'import pkgutil; print(str(list(pkgutil.iter_modules("/opt/python/2.7.12/lib/python2.7/site-packages"))))'
echo "========================="
python -c 'import numpy; import google; print(dir(google)); import google.protobuf; import pkgutil; print(str(list(pkgutil.iter_modules(google.protobuf.__path__)))); import google.protobuf.descriptor;  '
echo $PYTHON
echo PYTHONPATH: $PYTHONPATH
$PYTHON -c 'import numpy; import google.protobuf.descriptor; print("debug---------------")'
export PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/../../python/
$PYTHON -c 'import numpy; import google.protobuf.descriptor; print("debug---------------")'

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
