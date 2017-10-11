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

__all__ = ['define_py_data_sources2']


def define_py_data_source(file_list,
                          cls,
                          module,
                          obj,
                          args=None,
                          async=False,
                          data_cls=PyData):
    """
    Define a python data source.

    For example, the simplest usage in trainer_config.py as follow:

    ..  code-block:: python

        define_py_data_source("train.list", TrainData, "data_provider", "process")

    Or. if you want to pass arguments from trainer_config to data_provider.py, then

    ..  code-block:: python

        define_py_data_source("train.list", TrainData, "data_provider", "process",
                              args={"dictionary": dict_name})

    :param data_cls:
    :param file_list: file list name, which contains all data file paths
    :type file_list: basestring
    :param cls: Train or Test Class.
    :type cls: TrainData or TestData
    :param module: python module name.
    :type module: basestring
    :param obj: python object name. May be a function name if using
                PyDataProviderWrapper.
    :type obj: basestring
    :param args: The best practice is using dict to pass arguments into
                 DataProvider, and use :code:`@init_hook_wrapper` to
                 receive arguments.
    :type args: string or picklable object
    :param async: Load Data asynchronously or not.
    :type async: bool
    :return: None
    :rtype: None
    """
    if isinstance(file_list, list):
        file_list_name = 'train.list'
        if cls == TestData:
            file_list_name = 'test.list'
        with open(file_list_name, 'w') as f:
            f.writelines(file_list)
        file_list = file_list_name

    if not isinstance(args, basestring) and args is not None:
        args = pickle.dumps(args, 0)

    cls(
        data_cls(
            files=file_list,
            load_data_module=module,
            load_data_object=obj,
            load_data_args=args,
            async_load_data=async))


def define_py_data_sources(train_list,
                           test_list,
                           module,
                           obj,
                           args=None,
                           train_async=False,
                           data_cls=PyData):
    """
    The annotation is almost the same as define_py_data_sources2, except that
    it can specific train_async and data_cls.

    :param data_cls:
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
    :param train_async: Is training data load asynchronously or not.
    :type train_async: bool
    :return: None
    :rtype: None
    """

    def __is_splitable__(o):
        return (isinstance(o, list) or
                isinstance(o, tuple)) and hasattr(o, '__len__') and len(o) == 2

    assert train_list is not None or test_list is not None
    assert module is not None and obj is not None

    test_module = module
    train_module = module
    if __is_splitable__(module):
        train_module, test_module = module

    test_obj = obj
    train_obj = obj
    if __is_splitable__(obj):
        train_obj, test_obj = obj

    if args is None:
        args = ""

    train_args = args
    test_args = args
    if __is_splitable__(args):
        train_args, test_args = args

    if train_list is not None:
        define_py_data_source(train_list, TrainData, train_module, train_obj,
                              train_args, train_async, data_cls)

    if test_list is not None:
        define_py_data_source(test_list, TestData, test_module, test_obj,
                              test_args, False, data_cls)


def define_py_data_sources2(train_list, test_list, module, obj, args=None):
    """
    Define python Train/Test data sources in one method. If train/test use
    the same Data Provider configuration, module/obj/args contain one argument,
    otherwise contain a list or tuple of arguments. For example\:

    ..  code-block:: python

        define_py_data_sources2(train_list="train.list",
                                test_list="test.list",
                                module="data_provider"
                                # if train/test use different configurations,
                                # obj=["process_train", "process_test"]
                                obj="process",
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

    def py_data2(files, load_data_module, load_data_object, load_data_args,
                 **kwargs):
        data = create_data_config_proto()
        data.type = 'py2'
        data.files = files
        data.load_data_module = load_data_module
        data.load_data_object = load_data_object
        data.load_data_args = load_data_args
        data.async_load_data = False
        return data

    define_py_data_sources(
        train_list=train_list,
        test_list=test_list,
        module=module,
        obj=obj,
        args=args,
        data_cls=py_data2)
