// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPARSE_H
#define SPARSE_H

#include <cuda_runtime.h>
#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * sparseAllGReduce
 *
 * Each device gathers sparsecount encode values from other GPUs into
 * gatherbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes gatherbuff count is equal to nranks*sparsecount, which means that
 * gatherbuff count
 * should have a size of at least nranks*sparsecount elements.
 *
 * In-place operations will happen if encodebuff == gatherbuff + rank *
 * sparsecount.
 */
void sparseAllGReduce(const void* encodebuff, void* gatherbuff,
                      const int sparsecount, void* recvbuff,
                      const int recvcount, ncclDataType_t datatype,
                      ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif
