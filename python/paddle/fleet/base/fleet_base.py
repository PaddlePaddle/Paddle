#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
from paddle.fleet import RoleMakerBase
from . import obj_creator

__all__ = ['Fleet']


class Fleet(object):
    """
    Fleet is the base class, transpiler and pslib are implementation of Fleet.

    Args:
        mode(Mode): the implementation of Fleet's mode.

    Returns:
        None
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.util_obj = None
        self._fleet_obj = None
        self._role_maker_obj = None

    def is_first_worker(self):
        """
        Check whether the node is the first instance of worker.

        Returns:
            bool: True if this is the first node of worker,
                  False if not.
        """
        return self._role_maker.is_first_worker()

    def worker_index(self):
        """
        Get current worker index.

        Returns:
            int: node id
        """
        return self._role_maker.worker_index()

    def worker_num(self):
        """
        Get current total worker number.

        Returns:
            int: worker numbers
        """
        return self._role_maker.worker_num()

    def is_worker(self):
        """
        Check whether the node is an instance of worker.

        Returns:
            bool: True if this is a node of worker,
                  False if not.
        """
        return self._role_maker.is_worker()

    def worker_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_trainer_endpoints())
        else:
            return self._role_maker.get_trainer_endpoints()

    def server_num(self):
        """
        Get current total worker number.

        Returns:
            int: server number
        """
        return len(self._role_maker.get_pserver_endpoints())

    def server_index(self):
        """
        Get current server index.

        Returns:
            int: node id
        """
        return self._role_maker.server_index()

    def server_endpoints(self, to_string=False):
        """
        Get current server endpoints, such as ["127.0.0.1:1001", "127.0.0.1:1002"].

        Returns:
            list/string: server endpoints
        """

        if to_string:
            return ",".join(self._role_maker.get_pserver_endpoints())
        else:
            return self._role_maker.get_pserver_endpoints()

    def is_server(self):
        """
        Check whether the node is an instance of server.

        Returns:
            bool: True if this is a node of server,
                  False if not.
        """
        return self._role_maker.is_server()

    @property
    def util(self):
        """
        less commonly used api and task specific api
        should be in util
        """
        return self._util

    def init(self, role_maker=None):
        """
        should be called only once in user's python scripts,
        init() will initialize RoleMaker which is used for identifying
            current node's role, e.g. worker, server, etc.

        Args:
            role_maker(RoleMakerBase): subclass of RoleMakerBase.

        Returns:
            None
        """
        if role_maker and not isinstance(role_maker, RoleMakerBase):
            raise TypeError("role_maker must be an instance of RoleMakerBase")

        self._role_maker = role_maker
        self._role_maker.generate_role()

        self._fleet_obj = obj_creator._create_fleet_obj_from_role_maker(
            role_maker)
        self._util_obj = obj_creator._create_fleet_util_from_role_maker(
            role_maker)

    def barrier_worker(self):
        """
        barrier between workers
        """
        self._role_maker.barrier_worker()

    @abc.abstractmethod
    def init_worker(self):
        pass

    @abc.abstractmethod
    def init_server(self, model_dir=None):
        pass

    @abc.abstractmethod
    def run_server(self):
        pass

    @abc.abstractmethod
    def stop_worker(self):
        pass

    def distributed_optimizer(self, optimizer, strategy=None):
        assert self._fleet_obj is not None
        assert self._util is not None
        self._fleet_obj.distributed_optimizer(optimizer, strategy)

    def save_inference_model(self,
                             executor,
                             dirname,
                             feeded_var_names,
                             target_vars,
                             main_program=None,
                             export_for_deployment=True):
        assert self._fleet_obj is not None
        assert self._util is not None
        self._fleet_obj.save_inference_model(
            executor, dirname, feeded_var_names, target_vars, main_program,
            export_for_deployment)

    def save_persistables(self, executor, dirname, main_program=None):
        assert self._fleet_obj is not None
        assert self._util is not None
        self._fleet_obj.save_persistables(executor, dirname, main_program)
