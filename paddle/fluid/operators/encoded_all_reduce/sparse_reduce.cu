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

#include "common.h"
#include "reduce_kernel.h"
#include "sparse_reduce.h"

#define FUNC_INDEX(redop, dtype) ((redop * ncclNumTypes) + dtype)

#define FUNCS3(op, ctype) sparseReduceKernel<op<ctype>, ctype>

#define FUNCS2(op)                                                  \
  FUNCS3(op, int8_t)                                                \
  , FUNCS3(op, uint8_t), FUNCS3(op, int32_t), FUNCS3(op, uint32_t), \
      FUNCS3(op, int64_t), FUNCS3(op, uint64_t), FUNCS3(op, half),  \
      FUNCS3(op, float), FUNCS3(op, double)

#define FUNCS() \
  FUNCS2(FuncSum), FUNCS2(FuncProd), FUNCS2(FuncMin), FUNCS2(FuncMax)

#define THREADS 256
#define BLOCKS 32
#define DIVUP(x, y) (((x) + (y)-1) / (y))

template <class FUNC, typename T>
__global__ void sparseReduceKernel(void* gather_buff, const int sparse_count,
                                   void* recv_buff, const int recv_count,
                                   const int ranks) {
  const int tid = threadIdx.x + blockIdx.x * THREADS;
  const int N = gridDim.x * THREADS;

  int* index = (int*)gather_buff;
  T* value = (T*)((char*)gather_buff + sizeof(uint32_t) * sparse_count);
  T* dst = (T*)recv_buff;
  uint32_t idx;
  T val;
  for (int j = tid; j < sparse_count; j += N) {
    idx = index[tid];
    val = value[tid];
    val = FUNC()(val, dst[idx]);
    dst[idx] = val;
  }
}

// IMPL template, store all func point
typedef void (*Kern_t)(void*, const int, void*, const int, const int);
static const Kern_t Funcs[ncclNumOps * ncclNumTypes] = {FUNCS()};

void sparseReduce(void* gatherbuff, const int sparsecount, void* recvbuff,
                  const int recvcount, const int ranks, ncclDataType_t datatype,
                  ncclRedOp_t op, cudaStream_t stream) {
  // grid sync, Todo. use cooperative groups for grid sync
  for (int i = 0; i < ranks; ++i) {
    Funcs[FUNC_INDEX(op, datatype)]<<<min(DIVUP(sparsecount, THREADS), BLOCKS),
                                      THREADS, 0, stream>>>(
        gatherbuff, sparsecount, recvbuff, recvcount, ranks);
    gatherbuff =
        (void*)((char*)gatherbuff +
                (sizeof(uint32_t) + ncclTypeSize(datatype)) * sparsecount);
  }
}
