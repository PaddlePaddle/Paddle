# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

DISTRIBUTED_OPERATORS = {}


class DistributedOperator:
    def __init__(self):
        self._impls = []
        self._name = None

    def register_impl(self, dist_impl):
        self._impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self._impls[impl_idx]

    def get_impls(self):
        return self._impls


class DistributedOperatorImpl:
    def __init__(self):
        self._name = None

    def forward(self, dist_ctx, *args, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def backward(self, dist_ctx, *grad_outputs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def get_name(self):
        return self._name

    def is_process_mesh_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def is_input_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def is_output_compatible(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def is_compatible(self, op_dist_attr):
        return self.is_process_mesh_compatible(op_dist_attr) \
            and self.is_input_compatible(op_dist_attr) \
            and self.is_output_compatible(op_dist_attr)

    def update_dims_mapping(self, op_dist_attr):
        raise NotImplementedError("Please Implement this method in Subclass.")


def register_distributed_operator(name, dist_op):
    global DISTRIBUTED_OPERATORS
    DISTRIBUTED_OPERATORS[name] = dist_op


def get_distributed_operator(name):
    global DISTRIBUTED_OPERATORS
    return DISTRIBUTED_OPERATORS.get(name, None)


def register_distributed_operator_impl(name, dist_impl):
    dist_op = get_distributed_operator(name)
    if dist_op is not None:
        dist_op.register_impl(dist_impl)
    else:
        assert False, "Must register distributed operator first."


def get_distributed_operator_impl(name, impl_idx):
    global DISTRIBUTED_OPERATORS
    return DISTRIBUTED_OPERATORS[name].get_impl(impl_idx)


def find_best_compatible_distributed_operator_impl(name, op_dist_attr,
                                                   fwd=True):
    """
    Here just return the first compatible implemention. 
    This will be improved by cost model in the future.
    """
    dist_op = get_distributed_operator(name)
    if dist_op is None:
        return None, -1
    compatible_impls = []
    impls = dist_op.get_impls()
    if fwd:
        for idx, impl in enumerate(impls):
            if impl.is_process_mesh_compatible(op_dist_attr) \
                and impl.is_input_compatible(op_dist_attr):
                compatible_impls.append((impl, idx))
    else:
        for idx, impl in enumerate(impls):
            if impl.is_process_mesh_compatible(op_dist_attr) \
                and impl.is_output_compatible(op_dist_attr):
                compatible_impls.append((impl, idx))

    if compatible_impls:
        best_compatible_impl, idx = compatible_impls[0]
    else:
        best_compatible_impl, idx = None, -1

    return best_compatible_impl, idx
