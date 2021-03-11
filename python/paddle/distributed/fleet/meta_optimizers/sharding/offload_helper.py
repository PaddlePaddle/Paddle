# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ..common import is_optimizer_op, OP_ROLE_KEY, OpRole
from paddle.fluid import unique_name


class OffloadHelper(object):
    cpu_place_type = 0
    cuda_place_type = 1
    cuda_pinned_place_type = 2

    def __init__(self):
        pass
        "0: dst is on CPUPlace. "
        "1: dst is on CUDAPlace. "
        "2: dst is on CUDAPinnedPlace. "

    def _insert_memcpy_op(self, block, idx, src_name, dst_name, dst_place_type):
        src_var = block.var(src_name)
        dst_var = block.var(dst_name)
        block._insert_op_without_sync(
            idx,
            type='memcpy',
            inputs={'X': src_var},
            outputs={'Out': dst_var},
            attrs={
                'dst_place_type': dst_place_type,
                OP_ROLE_KEY: OpRole.Optimize,
            })

    def _insert_fetch_op(self, block, idx, src_name, dst_name):
        self._insert_memcpy_op(block, idx, src_name, dst_name,
                               OffloadHelper.cuda_place_type)

    def _insert_offload_op(self, block, idx, src_name, dst_name):
        self._insert_memcpy_op(block, idx, src_name, dst_name,
                               OffloadHelper.cuda_pinned_place_type)

    def _get_offload_var_name(self, name):
        return unique_name.generate(name + '@offload')

    def _create_offload_var(self, var_name, offload_var_name, blocks):
        for block in blocks:
            var = block.var(var_name)
            var.persistable = False
            offload_var = block.create_var(
                name=offload_var_name,
                shape=var.shape,
                dtype=var.dtype,
                persistable=True)

    def offload(self, block, startup_block):
        """
        (m1, m2) = prefetch(m1@offload, m2@offload)
        (m1out, m2out, pout) = adam(m1, m2, p)
        (m1@offload, m2@offload) = memcpy(m1, m2)
        """
        vars_name_to_offload_name = dict()

        # main_block add offload
        for idx, op in reversed(list(enumerate(block.ops))):
            if not is_optimizer_op(op):
                break

            vars_name = []
            if op.type == "adam":
                # {Moment1Out = [''], Moment2Out = [''], ParamOut = ['']} =
                # adam(inputs={Moment1 = [''], Moment2 = [''], Param = ['']})
                vars_name.append(op.desc.input("Moment1")[0])
                vars_name.append(op.desc.input("Moment2")[0])
            elif op.type == 'momentum':
                pass
            elif op.type == 'lars':
                pass
            elif op.type == 'lamb':
                pass

            # step1: create and init offload_var
            for var_name in vars_name:
                assert var_name not in vars_name_to_offload_name

                offload_var_name = self._get_offload_var_name(var_name)
                vars_name_to_offload_name[var_name] = offload_var_name

                self._create_offload_var(var_name, offload_var_name,
                                         [block, startup_block])

            # step2: insert offload op
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_offload_op(block, idx + 1, var_name,
                                        offload_var_name)

            # step3: insert fetch op
            for var_name in vars_name:
                offload_var_name = vars_name_to_offload_name[var_name]
                self._insert_fetch_op(block, idx, offload_var_name, var_name)

        # startup_block add offload
        visited_vars = set()
        for idx, op in reversed(list(enumerate(startup_block.ops))):
            for out_name in op.output_arg_names:
                if out_name in visited_vars:
                    continue

                if out_name in vars_name_to_offload_name:
                    var_name = out_name
                    offload_var_name = vars_name_to_offload_name[var_name]
                    # insert offload op after var is generated
                    self._insert_offload_op(startup_block, idx + 1, var_name,
                                            offload_var_name)
                visited_vars.add(out_name)
