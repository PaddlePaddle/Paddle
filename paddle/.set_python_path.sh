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

#
# A simple test driver for cmake. 
# set PYTHONPATH before run command.
# Usage:
#    ./.set_python_pash.sh -p YOUR_PYTHON_PATH {exec...}
# 
# It same as PYTHONPATH=${YOUR_PYTHON_PATH}:$PYTHONPATH {exec...}
#

if ! python -c "import paddle" >/dev/null 2>/dev/null; then
  PYPATH=""
  set -x
  while getopts "d:" opt; do
    case $opt in
      d)
        PYPATH=$OPTARG
        ;;
    esac
  done
  shift $(($OPTIND - 1))
  export PYTHONPATH=$PYPATH:$PYTHONPATH
  $@
else
  echo "paddle package is already in your PYTHONPATH. But unittest need a clean environment."
  echo "Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'"
  exit 1
fi
