// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_RCCL)
#include <hip/hip_runtime.h>

#include "paddle/phi/backends/dynload/rccl.h"

#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuFree hipFree
#else
#include <cuda_runtime.h>

#include "paddle/phi/backends/dynload/nccl.h"

#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuFree cudaFree
#endif

namespace phi::distributed {
void NCCLDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int64_t dtype) {
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(tensor.dtype()),
      dtype,
      common::errors::InvalidArgument(
          "Tensors in communication are expected to have the same data type."));
}

void NCCLDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int root_rank,
                                     int cur_rank,
                                     ncclComm_t comm) {
  constexpr int kSize = sizeof(int64_t);
  int64_t dtype_host = static_cast<int64_t>(tensor.dtype());
  int64_t* dtype_device;
  PADDLE_ENFORCE_GPU_SUCCESS(gpuMalloc(&dtype_device, kSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      gpuMemcpy(dtype_device, &dtype_host, kSize, gpuMemcpyHostToDevice));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclBroadcast(dtype_device,
                                                         dtype_device,
                                                         1,
                                                         ncclInt64,
                                                         root_rank,
                                                         comm,
                                                         kDefaultStream));

  if (root_rank == cur_rank) {
    VLOG(3) << "Dynamic check broadcast metadata, dtype: " << dtype_host;
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(
        gpuMemcpy(&dtype_host, dtype_device, kSize, gpuMemcpyDeviceToHost));
    VLOG(3) << "Dynamic check recv metadata, dtype: " << dtype_host;
    CheckDataType(tensor, dtype_host);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(gpuFree(dtype_device));
}

void NCCLDynamicCheck::CheckShape(const phi::DenseTensor& tensor,
                                  int64_t shape) {
  PADDLE_ENFORCE_EQ(
      tensor.numel(),
      shape,
      common::errors::InvalidArgument(
          "Tensors in communication are expected to have matching sizes."));
}

void NCCLDynamicCheck::CheckShape(const phi::DenseTensor& tensor,
                                  int root_rank,
                                  int cur_rank,
                                  ncclComm_t comm) {
  CheckDataType(tensor, root_rank, cur_rank, comm);

  constexpr int kSize = sizeof(int64_t);
  int64_t shape_host = tensor.numel();
  int64_t* shape_device;

  PADDLE_ENFORCE_GPU_SUCCESS(gpuMalloc(&shape_device, kSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      gpuMemcpy(shape_device, &shape_host, kSize, gpuMemcpyHostToDevice));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclBroadcast(shape_device,
                                                         shape_device,
                                                         1,
                                                         ncclInt64,
                                                         root_rank,
                                                         comm,
                                                         kDefaultStream));

  if (root_rank == cur_rank) {
    VLOG(3) << "Dynamic check broadcast metadata, shape: " << shape_host;
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(
        gpuMemcpy(&shape_host, shape_device, kSize, gpuMemcpyDeviceToHost));
    VLOG(3) << "Dynamic check recv metadata, shape: " << shape_host;
    CheckShape(tensor, shape_host);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(gpuFree(shape_device));
}

void NCCLDynamicCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  const std::vector<int64_t>& in_size_each_rank,
                                  int cur_rank,
                                  int world_size,
                                  ncclComm_t comm) {
  CheckDataType(out_tensor, /*root_rank*/ 0, cur_rank, comm);
  CheckDataType(in_tensor, /*root_rank*/ 0, cur_rank, comm);

  constexpr int kSize = sizeof(int64_t);
  int64_t in_row_size = in_tensor.numel() / in_tensor.dims()[0];

  for (int rank = 0; rank < world_size; ++rank) {
    int64_t in_shape_host = in_size_each_rank[rank] * in_row_size;
    int64_t* in_shape_device;
    PADDLE_ENFORCE_GPU_SUCCESS(gpuMalloc(&in_shape_device, kSize));
    PADDLE_ENFORCE_GPU_SUCCESS(gpuMemcpy(
        in_shape_device, &in_shape_host, kSize, gpuMemcpyHostToDevice));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclReduce(in_shape_device,
                                                        in_shape_device,
                                                        1,
                                                        ncclInt64,
                                                        ncclSum,
                                                        rank,
                                                        comm,
                                                        kDefaultStream));
    if (rank == cur_rank) {
      PADDLE_ENFORCE_GPU_SUCCESS(gpuMemcpy(
          &in_shape_host, in_shape_device, kSize, gpuMemcpyDeviceToHost));
      VLOG(3) << "Dynamic check recv metadata, shape: " << in_shape_host;
      CheckShape(out_tensor, in_shape_host);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(gpuFree(in_shape_device));
  }
}

void NCCLDynamicCheck::CheckGatherShape(
    const phi::DenseTensor& in_tensor,
    const std::vector<phi::DenseTensor>& out_tensors,
    int root_rank,
    int cur_rank,
    int world_size,
    ncclComm_t comm) {
  std::vector<int64_t> shapes(world_size, 0);
  shapes[cur_rank] = in_tensor.numel();
  int64_t* in_shape_device;
  PADDLE_ENFORCE_GPU_SUCCESS(
      gpuMalloc(&in_shape_device, world_size * sizeof(int64_t)));
  PADDLE_ENFORCE_GPU_SUCCESS(gpuMemcpy(in_shape_device,
                                       shapes.data(),
                                       world_size * sizeof(int64_t),
                                       gpuMemcpyHostToDevice));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(in_shape_device,
                                                         in_shape_device,
                                                         world_size,
                                                         ncclInt64,
                                                         ncclSum,
                                                         comm,
                                                         kDefaultStream));
  PADDLE_ENFORCE_GPU_SUCCESS(gpuMemcpy(shapes.data(),
                                       in_shape_device,
                                       world_size * sizeof(int64_t),
                                       gpuMemcpyDeviceToHost));
  PADDLE_ENFORCE_GPU_SUCCESS(gpuFree(in_shape_device));

  if (cur_rank == root_rank) {
    for (int i = 0; i < world_size; i++) {
      CheckShape(out_tensors[i], shapes[i]);
    }
  }
}
}  // namespace phi::distributed
