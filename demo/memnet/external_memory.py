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
    """
    External memory network class, with differentiable read/write heads.

    :param name: name for the external memory
    :type name:  basestring
    :param mem_slot_size: number of slots to be used for the external memory
    :type mem_slot_size: int
    :param mem_fea_size: size of each memory slot
    :type mem_fea_size: int
    :param is_test: flag indicating training (is_test=False) or testing (is_test=True)
    :type is_test: bool
    :param scale: a multiplicative factor applied to the read/write weights
    :param scale: int
    """
    def __init__(self, name, mem_slot_size, mem_fea_size, is_test=False, scale=5):
        self.name = name
        self.mem_slot_size = mem_slot_size
        self.mem_fea_size = mem_fea_size
        self.scale = scale
        self.external_memory = memory(name=self.name,                           	
                                      size=mem_fea_size*mem_slot_size,
                                      boot_bias= ParamAttr(initial_std=0.01,
                                                           initial_mean=0.))
        self.is_test = is_test

    def read(self, read_key):
        """
        Read head for the external memory. 
        :param read_key: key used for reading via content-based addressing, 
                         with size as mem_fea_size 
        :type read_key: LayerOutput
        :return: memory_read
        :rtype: LayerOutput 
        """
        cosine_similarity_read = cos_sim(read_key, self.external_memory, scale=self.scale, size=self.mem_slot_size)
        norm_cosine_similarity_read = mixed_layer(input=
                                                  identity_projection(cosine_similarity_read),
                                                  bias_attr = False,
                                                  act = SoftmaxActivation(),
                                                  size = self.mem_slot_size,
                                                  name=self.name+'_read_weight')

        memory_read = linear_comb_layer(weights=norm_cosine_similarity_read, 
                                        vectors=self.external_memory,
                                        size=self.mem_fea_size, name=self.name+'_read_content')

        if self.is_test:
            print_layer(input=[norm_cosine_similarity_read, memory_read])

        return memory_read

    def write(self, write_key):
        """
        Write head for the external memory. 
        :param write_key: the key (and content) used for writing via content-based addressing,
                          with size as mem_fea_size  
        :type write_key: LayerOutput
        :return: updated memory content
        :rtype: LayerOutput 
        """
        cosine_similarity_write = cos_sim(write_key, self.external_memory, 
                                          scale=self.scale, size=self.mem_slot_size)
        norm_cosine_similarity_write = mixed_layer(input=
                                                   identity_projection(cosine_similarity_write),
                                                   bias_attr = False,
                                                   act = SoftmaxActivation(),
                                                   size = self.mem_slot_size,
                                                   name=self.name+'_write_weight')
        if self.is_test:
            print_layer(input=[norm_cosine_similarity_write])
    
        add_vec = mixed_layer(input = full_matrix_projection(write_key),
                              bias_attr = None,
                              act = SoftmaxActivation(),
                              size = self.mem_fea_size,
                              name=self.name+'_add_vector')

        erase_vec = self.make_constant_vector(self.mem_fea_size, 1.0, write_key, self.name+"_constant_vector")

        if self.is_test:
            print_layer(input=[erase_vec])
            print_layer(input=[add_vec])

        out_prod = out_prod_layer(norm_cosine_similarity_write, erase_vec, name=self.name+"_outer")

        memory_remove = mixed_layer(input=dotmul_operator(a=self.external_memory, b=out_prod))

        memory_removed = self.external_memory - memory_remove

        out_prod_add = out_prod_layer(norm_cosine_similarity_write, add_vec, name=self.name+"_outer_add")
        memory_output = addto_layer(input=[memory_removed, out_prod_add], name=self.name)

        if self.is_test:
            print_layer(input=[memory_output])

        return memory_output

    def make_constant_vector(self, vec_size, value, dummy_input, layer_name):
        """
        Auxiliary function for generating a constant vector. 
        :param vec_size: the size of the constant vector 
        :type vec_size: int
        :param value: value of the elements in the constant vector
        :type value: float
        :param dummy_input: a dummy input layer to the constant vector network
        :type LayerOutput
        :param layer_name: name for the constant vector
        :type layer_name: basestring
        :return: memory_read
        :rtype: LayerOutput 
        """
        constant_scalar = mixed_layer(input=full_matrix_projection(input=dummy_input,
                                      param_attr = ParamAttr(learning_rate = 0, 
                                                             initial_mean = 0,
                                                             initial_std = 0)),
                                      bias_attr = ParamAttr(initial_mean=value, 
                                                            initial_std=0.0, 
                                                            learning_rate=0),
                                      act = LinearActivation(),
                                      size = 1,
                                      name = layer_name+'_constant_scalar')
        constant = mixed_layer(input=full_matrix_projection(input=constant_scalar,
                                      param_attr=ParamAttr(learning_rate = 0,
                                                           initial_mean = 1, 
                                                           initial_std = 0)),
                                      bias_attr = False,
                                      act = LinearActivation(),
                                      size = vec_size,
                                      name = layer_name)
        return constant


