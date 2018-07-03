#!/usr/bin/python
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
""" module for launching cluster job """

import os
import argparse
import socket
import copy
import time
import signal

from fabric.api import run, put, settings, env, prefix
from fabric.tasks import execute

#configuration for cluster
import conf


def refine_unknown_args(cmd_args):
    '''
    refine unknown parameters to handle some special parameters
    '''
    new_args = []
    for arg in cmd_args:
        if arg.startswith("--") and arg.find("=") != -1:
            equal_pos = arg.find("=")  #find first = pos
            arglist = list(arg)
            arglist[equal_pos] = " "
            arg = "".join(arglist)
            arg = arg.lstrip("-")
            new_args += arg.split(" ")
        elif arg.startswith("--") and arg.find("=") == -1:
            arg = arg.lstrip("-")
            new_args.append(arg)
        else:
            new_args.append(arg)
    return new_args


def kill_process():
    '''
    kill comments threads
    '''
    run("ps aux \
         | grep paddle_process_by_paddle \
         | grep -v grep  \
         | awk '{print $2}' \
         | xargs kill > /dev/null 2>&1")


def job_prepare(jobdir, data=None):
    '''
    prepare job related workspace data

    Assuming you already installed PaddlePaddle in all nodes which means
    PaddlePaddle related bins and dependencies libraries.
    Assuming the train/test data have already been installed.
    This function just prepare all related model and other resources
    needed at runtime.
    '''

    def job_create_workspace(jobdir, data=None):
        '''
        prepare job workspace, common file, etc.
        '''
        log = os.path.join(jobdir, "log")
        if data is not None:
            #create job dir
            run('rm ' + jobdir + ' -fr && ' + 'mkdir -p ' + jobdir)
            #push data and paddle bin
