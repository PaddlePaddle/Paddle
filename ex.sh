#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

TestDir='build/python/paddle/fluid/tests/unittests/dygraph_to_static/'
LogDir=$(pwd)'/log_dy2st_model_tests'

test_files=$(ls $TestDir | egrep 'test.*.py$')
# test_files=$(ls $TestDir | egrep 'test_bert.py$')
regex="(test.*)\.py$"

export GLOG_v=0
export FLAGS_new_executor_sequential_run=true
export FLAGS_enable_tracer=true
for f in $test_files
do 
    if [[ $f =~ $regex ]]
    then
        name="${BASH_REMATCH[1]}\$"
        echo $name" start"
        cd build
        ctest -VV -R $name > $LogDir/log_$name 2>&1 
        status=$?
        if [ $status -eq 0 ]  
        then
            echo $name" yes"
        else
            echo $name" failed" 
        fi
        cd ..
    else
        echo $f
    fi
done

unset GLOG_v
unset FLAGS_new_executor_sequential_run
unset FLAGS_enable_tracer

LogDir=$(pwd)'/log_dy2st_model_tests_no_trace'

test_files=$(ls $TestDir | egrep 'test.*.py$')
# test_files=$(ls $TestDir | egrep 'test_bert.py$')
regex="(test.*)\.py$"

export GLOG_v=0
export FLAGS_new_executor_sequential_run=true
export FLAGS_enable_tracer=false
for f in $test_files
do 
    if [[ $f =~ $regex ]]
    then
        name="${BASH_REMATCH[1]}\$"
        echo $name" start"
        cd build
        ctest -VV -R $name > $LogDir/log_$name 2>&1 
        status=$?
        if [ $status -eq 0 ]  
        then
            echo $name" yes"
        else
            echo $name" failed" 
        fi
        cd ..
    else
        echo $f
    fi
done

unset GLOG_v
unset FLAGS_new_executor_sequential_run
unset FLAGS_enable_tracer
