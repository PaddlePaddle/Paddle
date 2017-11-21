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
import os
import optimizer
import layer
import activation
import parameters
import trainer
import event
import data_type
import topology
import networks
import evaluator
from . import dataset
from . import reader
from . import plot
import attr
import op
import pooling
import inference
import networks
import minibatch
import plot
import image
import paddle.trainer.config_parser as cp

__all__ = [
    'default_startup_program',
    'default_main_program',
    'optimizer',
    'layer',
    'activation',
    'parameters',
    'init',
    'trainer',
    'event',
    'data_type',
    'attr',
    'pooling',
    'dataset',
    'reader',
    'topology',
    'networks',
    'infer',
    'plot',
    'evaluator',
    'image',
    'master',
]

cp.begin_parse()


def auto_set_cpu_env(trainer_count):
    '''Auto set CPU environment if have not set before.
       export KMP_AFFINITY, OMP_DYNAMIC according to the Hyper Threading status.
       export OMP_NUM_THREADS, MKL_NUM_THREADS according to trainer_count.
    '''
    import platform
    if platform.system() != "Linux" and platform.system() != "Darwin":
        return

    def set_env(key, value):
        '''If the key has not been set in the environment, set it with value.'''
        assert isinstance(key, str)
        assert isinstance(value, str)
        envset = os.environ.get(key)
        if envset is None:
            os.environ[key] = value

    def is_hyperthreading_enabled():
        '''If Hyper Threading is enabled.'''
        if platform.system() == "Linux":
            percore = os.popen(
                "lscpu |grep \"per core\"|awk -F':' '{print $2}'|xargs")
            return int(percore.read()) != 1
        elif platform.system() == "Darwin":
            physical = int(os.popen("sysctl hw.physicalcpu").read())
            logical = int(os.popen("sysctl hw.logicalcpu").read())
            return logical > physical
        else:
            # do not support on other platform yet
            return False

    def get_logical_processors():
        '''Get the logical processors number'''
        import platform
        if platform.system() == "Linux":
            processors = os.popen(
                "grep \"processor\" /proc/cpuinfo|sort -u|wc -l")
            return int(processors.read())
        elif platform.system() == "Darwin":
            processors = os.popen("sysctl hw.logicalcpu")
            return int(processors.read())
        else:
            # do not support on other platform yet
            return 1

    if is_hyperthreading_enabled():
        set_env("OMP_DYNAMIC", "true")
        set_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
    else:
        set_env("OMP_DYNAMIC", "false")
        set_env("KMP_AFFINITY", "granularity=fine,compact,0,0")
    processors = get_logical_processors()
    threads = processors / trainer_count
    threads = '1' if threads < 1 else str(threads)
    set_env("OMP_NUM_THREADS", threads)
    set_env("MKL_NUM_THREADS", threads)


def init(**kwargs):
    import py_paddle.swig_paddle as api
    args = []
    args_dict = {}
    # NOTE: append arguments if they are in ENV
    for ek, ev in os.environ.iteritems():
        if ek.startswith("PADDLE_INIT_"):
            args_dict[ek.replace("PADDLE_INIT_", "").lower()] = str(ev)

    args_dict.update(kwargs)
    # NOTE: overwrite arguments from ENV if it is in kwargs
    for key in args_dict.keys():
        args.append('--%s=%s' % (key, str(args_dict[key])))

    auto_set_cpu_env(kwargs.get('trainer_count', 1))

    if 'use_gpu' in kwargs:
        cp.g_command_config_args['use_gpu'] = kwargs['use_gpu']
    if 'use_mkldnn' in kwargs:
        cp.g_command_config_args['use_mkldnn'] = kwargs['use_mkldnn']
    assert 'parallel_nn' not in kwargs, ("currently 'parallel_nn' is not "
                                         "supported in v2 APIs.")

    api.initPaddle(*args)


infer = inference.infer
batch = minibatch.batch
