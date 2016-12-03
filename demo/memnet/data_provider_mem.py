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

from paddle.trainer.PyDataProvider2 import *
import numpy as np

########################### Parameters for Data Generation #################
gen_range = 8 # same as the size of the dictionary
#--------------- parameters for generating training data -------------------
# the sequence has a all-zero sub-vector in the beginning followed with a non-zero vector
# seq = [zero_sub_seq, non_zero_sub_seq]

# parameters for non_zero_sub_seq
seq_len = 10     # length of the non_zero_sub_seq
seq_len_min = 2  # minimum length if is_fixed_len is False; 
                 # seq_len will be used as the maximum length in this case, 
                 # i.e., the length will be sampled from [seq_len_min, seq_len]
# parameters for zero_sub_seq 
seq_len_pre = 10 
seq_len_pre_min = 2
# number of training data
sample_num = 1000

# -------------- parameters for generating testing data --------------------
seq_len_test = 10
seq_len_min_test = 3
seq_len_pre_test = 10
seq_len_pre_test_min = 2 
sample_num_test = 1


seq_len = max(seq_len, seq_len_min)

def gen_data(sample_number, gen_range, seq_len, seq_len_min, seq_len_pre, seq_len_pre_min, is_fixed_len = True):
    data = []
    
    if is_fixed_len:
        seq_len_actual = seq_len

    for i in range(0, sample_number):
        sample = []
        if not is_fixed_len:
            seq_len_actual = np.random.randint(seq_len_min, seq_len)
            seq_len_actual_pre = np.random.randint(seq_len_pre_min, seq_len_pre)
        sample0 = np.random.randint(1, gen_range, size=seq_len_actual)
        sample_pre = np.zeros(seq_len_actual_pre)
        sample_pre = sample_pre.astype(int)
        sample = np.concatenate([sample_pre, sample0])
        data.append([sample.tolist(), sample0[0]])

    return data

def gen_data_prefix(sample_number, gen_range, seq_len, seq_len_min, seq_len_pre, is_fixed_len = True):
    data = []

    if is_fixed_len:
        seq_len_actual = seq_len

    for i in range(0, sample_number):
        sample = []
        if not is_fixed_len:
            seq_len_actual = np.random.randint(seq_len)+1
            seq_len_actual = max(seq_len_actual, seq_len_min)
        sample = np.random.randint(gen_range, size=seq_len_actual)
        data.append([sample.tolist(), sample[1]])

    return data
 
   
data = gen_data(sample_num, gen_range, seq_len, seq_len_min, seq_len_pre, seq_len_pre_min, False)
data_test = gen_data(sample_num_test, gen_range, seq_len_test, seq_len_min_test, seq_len_pre_test, seq_len_pre_test_min, False)


@provider(input_types={"input_sequence" : integer_value_sequence(gen_range+1),
                       "ground_truth": integer_value(gen_range+1)})
def process_seq_train(settings, file_name):
    for d in data:
        yield {"input_sequence": d[0], 'ground_truth': d[1]} 


@provider(input_types={"input_sequence" : integer_value_sequence(gen_range+1),
                       "ground_truth": integer_value(gen_range+1)})
def process_seq_test(settings, file_name):
    for d in data_test:
        yield {"input_sequence": d[0], 'ground_truth': d[1]}
