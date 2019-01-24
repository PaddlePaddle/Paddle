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
#include "sparse.h"
#include "sparse_reduce.h"

// extern "C"
//__attribute__ ((visibility("default")))
// void sparseAllGReduce(const void* encodebuff, void* gatherbuff, const int
// sparsecount,
//                     void* recvbuff, const int recvcount, ncclDataType_t
//                     datatype,
//                     ncclRedOp_t op,  ncclComm_t comm, cudaStream_t stream);
extern "C" __attribute__((visibility("default"))) void sparseAllGReduce(
    const void* encodebuff, void* gatherbuff, const int sparsecount,
    void* recvbuff, const int recvcount, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllGather(encodebuff, gatherbuff,
                          sparsecount * (sizeof(int) + ncclTypeSize(datatype)),
                          ncclChar, comm, stream));
  int ranks = 0;
  NCCLCHECK(ncclCommCount(comm, &ranks));
  sparseReduce(gatherbuff, sparsecount, recvbuff, recvcount, ranks, datatype,
               op, stream);
}
