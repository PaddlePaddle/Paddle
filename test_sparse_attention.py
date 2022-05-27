# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import sys
import time
np.random.seed(100)

paddle.set_default_dtype('float32')

paddle.seed(100)

paddle.fluid.framework._disable_legacy_dygraph()
'''
batch_size = 16
############## [1, 1, 128, 768], nnz:524330 ##############
csr = paddle.nn.functional.dropout(paddle.ones([128, 128]),
                                   0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, batch_size, 129]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, batch_size, 8154]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, batch_size, 128, 768])
key = paddle.rand([1, batch_size, 128, 768])
value = paddle.rand([1, batch_size, 128, 768])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)

print(query.shape)
# [2, 8, 768, 512]
print(key.shape)
# [2, 8, 768, 512]
print(value.shape)
# [2, 8, 768, 512]
print(csr_offset.shape)
# [2, 8, 769]
print(csr_columns.shape)
# [2, 8, 295136]
'''
'''
############## [1, 8, 1024, 768], nnz:524330 ##############
csr = paddle.nn.functional.dropout(paddle.ones([1024, 1024]),
                                   0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, 16, 1025]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, 16, 524330]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, 16, 1024, 768])
key = paddle.rand([1, 16, 1024, 768])
value = paddle.rand([1, 16, 1024, 768])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)

print(query.shape)
# [1, 16, 1024, 768]
print(key.shape)
# [1, 16, 1024, 768]
print(value.shape)
# [1, 16, 1024, 768]
print(csr_offset.shape)
# [1, 16, 1025]
print(csr_columns.shape)
# [1, 16, 524330]


############## [1, 32, 1024, 768], nnz:524330 ##############
csr = paddle.nn.functional.dropout(paddle.ones([1024, 1024]),
                                   0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, 32, 1025]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, 32, 524330]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, 32, 1024, 768])
key = paddle.rand([1, 32, 1024, 768])
value = paddle.rand([1, 32, 1024, 768])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)

print(query.shape)
# [1, 32, 1024, 768]
print(key.shape)
# [1, 32, 1024, 768]
print(value.shape)
# [1, 32, 1024, 768]
print(csr_offset.shape)
# [1, 32, 1025]
print(csr_columns.shape)
# [1, 32, 524330]


############## [1, 64, 1024, 768], nnz:524330 ##############
csr = paddle.nn.functional.dropout(paddle.ones([1024, 1024]),
                                   0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, 64, 1025]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, 64, 524330]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, 64, 1024, 768])
key = paddle.rand([1, 64, 1024, 768])
value = paddle.rand([1, 64, 1024, 768])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)

print(query.shape)
# [1, 64, 1024, 768]
print(key.shape)
# [1, 64, 1024, 768]
print(value.shape)
# [1, 64, 1024, 768]
print(csr_offset.shape)
# [1, 64, 1025]
print(csr_columns.shape)
# [1, 64, 524330]
'''

############## [16, 16, 128/512, 16/32/64], nnz:3035 ##############
csr = paddle.nn.functional.dropout(paddle.ones([512, 512]), 0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, 1, 513]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, 1, 131271]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, 1, 512, 32]).cast('float16')
key = paddle.rand([1, 1, 512, 32]).cast('float16')
value = paddle.rand([1, 1, 512, 32]).cast('float16')

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)

print(query.shape)
# (32, 3, 100, 100)
print(key.shape)
# (32, 3, 100, 100)
print(value.shape)
# (32, 3, 100, 100)
print(csr_offset.shape)
# (32, 3, 101)
print(csr_columns.shape)
# (32, 3, 3035)

##########################################################
'''
############## [64, 1, 128, 768], nnz:8154 ##############
csr = paddle.nn.functional.dropout(paddle.ones([128, 128]), 0.5).to_sparse_csr()

csr_offset_data = csr.crows().expand([64, 1, 129]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([64, 1, 8154]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()


query = paddle.rand([64, 1, 128, 768])
key = paddle.rand([64, 1, 128, 768])
value = paddle.rand([64, 1, 128, 768])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)


print(query.shape)
# [64, 1, 128, 768]
print(key.shape)
# [64, 1, 128, 768]
print(value.shape)
# [64, 1, 128, 768]
print(csr_offset.shape)
# [64, 1, 129]
print(csr_columns.shape)
# [64, 1, 8154]

##########################################################
'''
'''
############## [16, 16, 16, 16], nnz:106 ##############
csr = paddle.nn.functional.dropout(paddle.ones([16, 16]), 0.6).to_sparse_csr()

csr_offset_data = csr.crows().expand([16, 16, 17]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([16, 16, 106]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()


query = paddle.rand([16, 16, 16, 16])
key = paddle.rand([16, 16, 16, 16])
value = paddle.rand([16, 16, 16, 16])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)


print(query.shape)
# [16, 16, 16, 16]
print(key.shape)
# [16, 16, 16, 16]
print(value.shape)
# [16, 16, 16, 16]
print(csr_offset.shape)
# [16, 16, 17]
print(csr_columns.shape)
# [16, 16, 106]
'''

##########################################################
'''
############## [1, 16, 1500, 35], nnz:225529 ##############
csr = paddle.nn.functional.dropout(paddle.ones([1500, 1500]), 0.9).to_sparse_csr()

csr_offset_data = csr.crows().expand([1, 16, 1501]).numpy().astype('int32')
csr_columns_data = csr.cols().expand([1, 16, 225529]).numpy().astype('int32')

paddle.fluid.framework._enable_legacy_dygraph()

query = paddle.rand([1, 16, 1500, 35])
key = paddle.rand([1, 16, 1500, 35])
value = paddle.rand([1, 16, 1500, 35])

csr_offset = paddle.to_tensor(csr_offset_data)
csr_columns = paddle.to_tensor(csr_columns_data)


print(query.shape)
# [1, 16, 1500, 35]
print(key.shape)
# [1, 16, 1500, 35]
print(value.shape)
# [1, 16, 1500, 35]
print(csr_offset.shape)
# [1, 16, 1501]
print(csr_columns.shape)
# [1, 16, 225529]

##########################################################
'''

paddle.device.cuda.synchronize()
start = time.time()
for i in range(1000):
    output = paddle.nn.functional.sparse_attention(query, key, value,
                                                   csr_offset, csr_columns)
paddle.device.cuda.synchronize()
end = time.time()

print(output)
print("time cost: ", end - start)
