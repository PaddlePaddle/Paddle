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
            put(data + "/*", jobdir)
            run("mkdir -p " + log)
        run('rm -fr ' + log + "/*")

    def set_nodefile(nodeid):
        '''
        create nodefile for later usage
        '''
        run('echo ' + str(nodeid) + ' > ' + jobdir + '/nodefile')

    execute(job_create_workspace, jobdir, data, hosts=conf.HOSTS)
    for i in xrange(len(conf.HOSTS)):
        execute(set_nodefile, i, hosts=conf.HOSTS[i])
    #clean rubbish caused by exception 
    with settings(warn_only=True):
        execute(kill_process, hosts=conf.HOSTS)


def job_pserver(jobdir, pids=None):
    '''
    start all pservers
    '''
    pargs = " --num_gradient_servers=" + str(len(conf.HOSTS))
    pargs += (" --nics=" + conf.PADDLE_NIC)
    pargs += " --port=" + str(conf.PADDLE_PORT)
    pargs += " --ports_num=" + str(conf.PADDLE_PORTS_NUM)
    #always start sparse pserver by default
    pargs += " --ports_num_for_sparse=" + str(conf.PADDLE_PORTS_NUM_FOR_SPARSE)
    pargs += " --comment=" + "paddle_process_by_paddle"

    def start_pserver(jobdir, pargs):
        '''
        start pserver process with fabric executor
        '''
        with prefix('export LD_LIBRARY_PATH=' + \
                conf.LD_LIBRARY_PATH + \
                ':$LD_LIBRARY_PATH'):
            program = 'paddle pserver'
            run('cd ' + jobdir + '; '  + \
                'GLOG_logtostderr=0 GLOG_log_dir="./log" ' + \
                'nohup ' + \
                program + " " + pargs + ' > ./log/server.log 2>&1 < /dev/null & ',
                pty=False)

    execute(start_pserver, jobdir, pargs, hosts=conf.HOSTS)


def job_trainer(jobdir, train_args_dict, pids=None):
    '''
    start paddle trainer
    '''
    args = " --num_gradient_servers=" + str(len(conf.HOSTS))
    args += " --nics=" + conf.PADDLE_NIC
    args += " --port=" + str(conf.PADDLE_PORT)
    args += " --ports_num=" + str(conf.PADDLE_PORTS_NUM)
    args += " --comment=" + "paddle_process_by_paddle"
    ip_string = ""
    for i in xrange(len(conf.HOSTS)):
        host = conf.HOSTS[i]
        left = host.find("@")
        right = host.find(':')
        left = 0 if left == -1 else left + 1
        right = len(host) if right == -1 else right
        ip_string += (socket.gethostbyname(host[left:right]) + ",")
    ip_string = ip_string.rstrip(",")
    args += " --pservers=" + ip_string

    args_ext = ""
    for key, value in train_args_dict.items():
        args_ext += (' --' + key + '=' + value)
    args += " " + args_ext

    def start_trainer(jobdir, args):
        '''
        start trainer process with fabric executor
        '''
        with prefix('export LD_LIBRARY_PATH=' + \
                conf.LD_LIBRARY_PATH + \
                ':$LD_LIBRARY_PATH'):
            program = 'paddle train'
            run('cd ' + jobdir + '; '  + \
                'GLOG_logtostderr=0 '
                'GLOG_log_dir="./log" '
                'nohup ' + \
                program + " " + args + " > ./log/train.log 2>&1 < /dev/null & ",
                pty=False)

    for i in xrange(len(conf.HOSTS)):
        train_args = copy.deepcopy(args)
        train_args += " --trainer_id=" + str(i)
        execute(start_trainer, jobdir, train_args, hosts=conf.HOSTS[i])


def job_all(job_package, jobdir=None, train_args_dict=None):
    '''
    param job_package
    param train_args_dict
    '''
    if jobdir is None:
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        jobdir = conf.ROOT_DIR + "/JOB" + timestamp
    job_prepare(jobdir, job_package)
    job_pserver(jobdir)
    time.sleep(5)  #wait until pservers completely start
    job_trainer(jobdir, train_args_dict)
    job_clean()


def job_clean():
    '''
    if starting job failed from paddle internal, the framework always
    is launched successfully since these process are daemon processes.
    so this job_clean can alway clean job rubbish process with ctrl+c.
    '''

    def signal_handler(signal, frame):
        '''
        SIGINT handler
        '''

        def kill_process():
            run("ps aux \
                  | grep paddle_process_by_paddle \
                  | grep -v grep  \
                  | awk '{print $2}' \
                  | xargs kill > /dev/null 2>&1")

        with settings(warn_only=True):
            execute(kill_process, hosts=conf.HOSTS)

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="paddle.py", description='simple tool for cluster training')
    parser.add_argument(
        '-j',
        '--job_workspace',
        required=False,
        default=None,
        help='job workspace')
    parser.add_argument(
        '-p',
        '--job_dispatch_package',
        required=False,
        default=None,
        help='job package for dispatching to all other nodes')

    args, train_args_list = parser.parse_known_args()
    train_args = refine_unknown_args(train_args_list)
    train_args_dict = dict(zip(train_args[:-1:2], train_args[1::2]))

    if args.job_workspace is not None:
        #if assigned workspace, do not need to dispatch data,
        #so job_local_package should be None
        assert args.job_dispatch_package is None
        job_all(None, args.job_workspace, train_args_dict)
    elif args.job_dispatch_package is not None:
        assert args.job_workspace is None
        assert os.path.isdir(args.job_dispatch_package)
        job_all(args.job_dispatch_package, None, train_args_dict)
    else:
        print "--job_workspace or --job_dispatch_package should be set"
