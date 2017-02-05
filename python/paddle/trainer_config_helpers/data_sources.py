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
"""
Data Sources are helpers to define paddle training data or testing data.
"""
from paddle.trainer.config_parser import *
from .utils import deprecated

try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = ['setup_data_provider']


def setup_data_provider(train_list, test_list, module, function, args=None):
    """
    Define python Train/Test data sources in one method. If train/test use
    the same Data Provider configuration, module/obj/args contain one argument,
    otherwise contain a list or tuple of arguments. For example\:

    ..  code-block:: python

        setup_data_provider("train.list",
                            "test.list",
                            "data_provider"
                            "process",
                            args={"dictionary": dict_name})

    The related data provider can refer to :ref:`api_pydataprovider2_sequential_model` .

    :param train_list: Train list name.
    :type train_list: basestring
    :param test_list: Test list name.
    :type test_list: basestring
    :param module: python module name. If train and test is different, then
                   pass a tuple or list to this argument.
    :type module: basestring or tuple or list
    :param obj: python object name. May be a function name if using
                PyDataProviderWrapper. If train and test is different, then pass
                a tuple or list to this argument.
    :type obj: basestring or tuple or list
    :param args: The best practice is using dict() to pass arguments into
                 DataProvider, and use :code:`@init_hook_wrapper` to receive
                 arguments. If train and test is different, then pass a tuple
                 or list to this argument.
    :type args: string or picklable object or list or tuple.
    :return: None
    :rtype: None
    """

    def create_py_data_config_proto(list_file, module, function, args, **kwargs):
        proto = create_data_config_proto()
        proto.type = 'py2'
        proto.files = list_file
        proto.load_data_module = module
        proto.load_data_object = object
        proto.load_data_args = args
        proto.async_load_data = True
        return proto

    set_global_training_data_config_proto(
        create_py_data_config_proto(train_list, module, function, args))

    set_global_testing_data_config_proto(
        create_py_data_config_proto(test_list, module, function, args))
