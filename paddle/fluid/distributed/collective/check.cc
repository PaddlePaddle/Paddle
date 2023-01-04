// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/check.h"

#include "paddle/fluid/distributed/collective/nccl_tools.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/errors.h"

#ifdef PADDLE_WITH_HIP
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuFree hipFree
#else
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuFree cudaFree
#endif

namespace paddle {
namespace distributed {

// static checks
void CommStaticCheck::CheckRank(int rank, int world_size) {
  PADDLE_ENFORCE_GE(rank,
                    0,
                    phi::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      phi::errors::InvalidArgument("Rank is out of the process group."));
}

void CommStaticCheck::CheckPlace(const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(tensor.place()),
      true,
      platform::errors::InvalidArgument("Tensor should be in GPU place."));
}

void CommStaticCheck::CheckPlace(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor) {
  CheckPlace(out_tensor);
  CheckPlace(in_tensor);
  PADDLE_ENFORCE_EQ(
      out_tensor.place(),
      in_tensor.place(),
      phi::errors::InvalidArgument(
          "Input and output tensors should be on the same place."));
}

void CommStaticCheck::CheckDataType(const phi::DenseTensor& out_tensor,
                                    const phi::DenseTensor& in_tensor) {
  PADDLE_ENFORCE_EQ(
      out_tensor.dtype(),
      in_tensor.dtype(),
      phi::errors::InvalidArgument(
          "Input and output tensors should have the same data type."));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_GT(
      tensor.numel(),
      0,
      phi::errors::InvalidArgument("Size of tensor should be greater than 0."));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 int out_size_factor,
                                 int in_size_factor) {
  CheckShape(out_tensor);
  CheckShape(in_tensor);
  int64_t out_size = out_tensor.numel(), in_size = in_tensor.numel();
  PADDLE_ENFORCE_EQ(
      out_size * out_size_factor,
      in_size * in_size_factor,
      phi::errors::InvalidArgument(
          "Input and output tensors should have matching sizes."));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 int dst_rank,
                                 int cur_rank,
                                 int world_size,
                                 int out_size_factor,
                                 int in_size_factor) {
  CheckRank(dst_rank, world_size);
  CheckRank(cur_rank, world_size);

  CheckPlace(out_tensor, in_tensor);
  CheckDataType(out_tensor, in_tensor);

  if (dst_rank == cur_rank) {
    CheckShape(out_tensor, in_tensor, out_size_factor, in_size_factor);
  } else {
    CheckShape(out_tensor);
    CheckShape(in_tensor);
  }
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& tensor,
                                 int rank,
                                 int world_size) {
  CheckPlace(tensor);
  CheckRank(rank, world_size);
}

void CommStaticCheck::SameShape(const phi::DenseTensor& out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int dst_rank,
                                int cur_rank,
                                int world_size) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ 1,
             /*in_size_factor*/ 1);
}

void CommStaticCheck::ScatterLikeShape(const phi::DenseTensor& out_tensor,
                                       const phi::DenseTensor& in_tensor,
                                       int dst_rank,
                                       int cur_rank,
                                       int world_size) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ world_size,
             /*in_size_factor*/ 1);
}

void CommStaticCheck::GatherLikeShape(const phi::DenseTensor& out_tensor,
                                      const phi::DenseTensor& in_tensor,
                                      int dst_rank,
                                      int cur_rank,
                                      int world_size) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ 1,
             /*in_size_factor*/ world_size);
}

// dynamic checks
void CommDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int64_t dtype) {
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(tensor.dtype()),
      dtype,
      phi::errors::InvalidArgument(
          "Tensors in communication are expected to have the same data type."));
}

void CommDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int root_rank,
                                     int cur_rank,
                                     ncclComm_t comm) {
  constexpr int kSize = sizeof(int64_t);
  int64_t dtype_host = static_cast<int64_t>(tensor.dtype());
  int64_t* dtype_device;
  PADDLE_ENFORCE_GPU_SUCCESS(gpuMalloc(&dtype_device, kSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      gpuMemcpy(dtype_device, &dtype_host, kSize, gpuMemcpyHostToDevice));

  NCCL_CHECK(phi::dynload::ncclBroadcast(dtype_device,
                                         dtype_device,
                                         kSize,
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

void CommDynamicCheck::CheckShape(const phi::DenseTensor& tensor,
                                  int64_t shape) {
  PADDLE_ENFORCE_EQ(
      tensor.numel(),
      shape,
      phi::errors::InvalidArgument(
          "Tensors in communication are expected to have matching sizes."));
}

void CommDynamicCheck::CheckShape(const phi::DenseTensor& tensor,
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

  NCCL_CHECK(phi::dynload::ncclBroadcast(shape_device,
                                         shape_device,
                                         kSize,
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

void CommDynamicCheck::CheckShape(const phi::DenseTensor& out_tensor,
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

    NCCL_CHECK(phi::dynload::ncclReduce(in_shape_device,
                                        in_shape_device,
                                        kSize,
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

}  //  namespace distributed
}  //  namespace paddle
