#edit-mode: -*- python -*-
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from paddle.trainer_config_helpers import *


class ExternalMemory(object):
    def __init__(self, name, mem_slot_size, mem_fea_size, ext_mem_initial, is_test=False, scale=5):
        self.name = name
        self.mem_slot_size = mem_slot_size
        self.mem_fea_size = mem_fea_size
        self.scale = 5
        self.external_memory = memory(name=self.name,
                               	size=mem_fea_size*mem_slot_size,
                                boot_bias= ParamAttr(initial_std=0.01,
                                              initial_mean=0.))
        self.is_test = is_test

    def read(self, read_key):
        cosine_similarity_read = cos_sim(read_key, self.external_memory, scale=self.scale, size=self.mem_slot_size)
        norm_cosine_similarity_read = mixed_layer(input=
                                                  identity_projection(cosine_similarity_read),
                                                  bias_attr = False,
                                                  act = SoftmaxActivation(),
                                                  size = self.mem_slot_size,
                                                  name='read_weight')

        memory_read = linear_comb_layer(weights=norm_cosine_similarity_read, 
                                        vectors=self.external_memory,
                                        size=self.mem_fea_size, name='read_content')

        if self.is_test:
            print_layer(input=[norm_cosine_similarity_read, memory_read])

        return memory_read

    def write(self, write_key):
        cosine_similarity_write = cos_sim(write_key, self.external_memory, 
                                          scale=self.scale, size=self.mem_slot_size)
        norm_cosine_similarity_write = mixed_layer(input=
                                                   identity_projection(cosine_similarity_write),
                                                   bias_attr = False,
                                                   act = SoftmaxActivation(),
                                                   size = self.mem_slot_size,
                                                   name='write_weight')
        if self.is_test:
            print_layer(input=[norm_cosine_similarity_write])
    
        add_vec = mixed_layer(input = full_matrix_projection(write_key),
                              bias_attr = None,
                              act = SoftmaxActivation(),
                              size = self.mem_fea_size,
                              name='add_vector')

        erase_vec = self.MakeConstantVector(self.mem_fea_size, 1.0, write_key)


        if self.is_test:
            print_layer(input=[erase_vec])
            print_layer(input=[add_vec])

        out_prod = out_prod_layer(norm_cosine_similarity_write, erase_vec, name="outer")

        memory_remove = mixed_layer(input=dotmul_operator(a=self.external_memory, b=out_prod))

        memory_remove_neg = slope_intercept_layer(input=memory_remove, slope=-1.0, intercept=0)

        # memory_updated = memory_mat - memory_remove  = memory_mat + memory_remove_neg
        memory_removed = mixed_layer(input = [identity_projection(input=self.external_memory),
                                              identity_projection(input=memory_remove_neg)],
                                     bias_attr = False,
                                     act = LinearActivation())

        out_prod_add = out_prod_layer(norm_cosine_similarity_write, add_vec, name="outer_add")

        memory_output = mixed_layer(input = [identity_projection(input=memory_removed),
                                             identity_projection(input=out_prod_add)],
                                    bias_attr = False,
                                    act = LinearActivation(),
                                    name=self.name)
        if self.is_test:
            print_layer(input=[memory_output])

        return memory_output

    def MakeConstantVector(self, vec_size, value, dummy_input):
        constant_scalar = mixed_layer(input=full_matrix_projection(input=dummy_input,
                                      param_attr = ParamAttr(learning_rate = 0, 
                                                             initial_mean = 0,
                                                             initial_std = 0)),
                                      bias_attr = ParamAttr(initial_mean=value, 
                                                            initial_std=0.0, 
                                                            learning_rate=0),
                                      act = LinearActivation(),
                                      size = 1,
                                      name = 'constant_scalar')
        constant = mixed_layer(input=full_matrix_projection(input=constant_scalar,
                                      param_attr=ParamAttr(learning_rate = 0,
                                                           initial_mean = 1, 
                                                           initial_std = 0)),
                                      bias_attr = False,
                                      act = LinearActivation(),
                                      size = vec_size,
                                      name = 'constant_vector')
        return constant


