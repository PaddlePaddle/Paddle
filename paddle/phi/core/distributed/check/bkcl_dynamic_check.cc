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

#include "paddle/phi/core/distributed/check/bkcl_dynamic_check.h"

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/string_helper.h"

namespace phi::distributed {

void BKCLDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int64_t dtype) {
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(tensor.dtype()),
      dtype,
      common::errors::InvalidArgument(
          "Tensors in communication are expected to have the same data type."));
}

void BKCLDynamicCheck::CheckDataType(const phi::DenseTensor& tensor,
                                     int root_rank,
                                     int cur_rank,
                                     BKCLContext_t comm) {
  constexpr int kSize = sizeof(int64_t);
  int64_t dtype_host = static_cast<int64_t>(tensor.dtype());
  int64_t* dtype_device;
  PADDLE_ENFORCE_XPU_SUCCESS(
      xpu_malloc(reinterpret_cast<void**>(&dtype_device), kSize));
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(
      dtype_device, &dtype_host, kSize, XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_broadcast(
      comm, dtype_device, dtype_device, 1, BKCL_INT64, root_rank, 0));

  if (root_rank == cur_rank) {
    VLOG(3) << "Dynamic check broadcast metadata, dtype: " << dtype_host;
  } else {
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(
        &dtype_host, dtype_device, kSize, XPUMemcpyKind::XPU_DEVICE_TO_HOST));
    VLOG(3) << "Dynamic check recv metadata, dtype: " << dtype_host;
    CheckDataType(tensor, dtype_host);
  }
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_free(dtype_device));
}

void BKCLDynamicCheck::CheckShape(const phi::DenseTensor& tensor,
                                  int64_t shape) {
  PADDLE_ENFORCE_EQ(
      tensor.numel(),
      shape,
      common::errors::InvalidArgument(
          "Tensors in communication are expected to have matching sizes."));
}

void BKCLDynamicCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  const std::vector<int64_t>& in_size_each_rank,
                                  int cur_rank,
                                  int world_size,
                                  BKCLContext_t comm) {
  CheckDataType(out_tensor, /*root_rank*/ 0, cur_rank, comm);
  CheckDataType(in_tensor, /*root_rank*/ 0, cur_rank, comm);

  constexpr int kSize = sizeof(int64_t);
  int64_t in_row_size =
      in_tensor.dims()[0] == 0 ? 0 : in_tensor.numel() / in_tensor.dims()[0];

  for (int rank = 0; rank < world_size; ++rank) {
    int64_t in_shape_host = in_size_each_rank[rank] * in_row_size;
    int64_t* in_shape_device;
    PADDLE_ENFORCE_XPU_SUCCESS(
        xpu_malloc(reinterpret_cast<void**>(&in_shape_device), kSize));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(in_shape_device,
                                          &in_shape_host,
                                          kSize,
                                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_reduce(comm,
                                            in_shape_device,
                                            in_shape_device,
                                            1,
                                            BKCL_INT64,
                                            BKCL_ADD,
                                            rank,
                                            0));
    if (rank == cur_rank) {
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(&in_shape_host,
                                            in_shape_device,
                                            kSize,
                                            XPUMemcpyKind::XPU_DEVICE_TO_HOST));
      VLOG(3) << "Dynamic check recv metadata, shape: " << in_shape_host;
      CheckShape(out_tensor, in_shape_host);
    }
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_free(in_shape_device));
  }
}

void BKCLDynamicCheck::CheckAlltoAllShape(
    const std::vector<phi::DenseTensor>& out_tensor,
    const std::vector<phi::DenseTensor>& in_tensor,
    int cur_rank,
    int world_size,
    BKCLContext_t comm) {
  int64_t first_dtype = static_cast<int64_t>(in_tensor[0].dtype());
  constexpr int kSize = sizeof(int64_t);
  CheckDataType(in_tensor[0], /*root_rank*/ 0, cur_rank, comm);
  for (int rank = 0; rank < world_size; ++rank) {
    CheckDataType(in_tensor[rank], first_dtype);
    CheckDataType(out_tensor[rank], first_dtype);

    int64_t in_shape_host = in_tensor[rank].numel();
    int64_t* in_shape_device;
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_malloc(
        reinterpret_cast<void**>(&in_shape_device), kSize * world_size));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(in_shape_device + cur_rank,
                                          &in_shape_host,
                                          kSize,
                                          XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_all_gather(
        comm, in_shape_device + cur_rank, 1, in_shape_device, BKCL_INT64, 0));
    if (rank == cur_rank) {
      std::vector<int64_t> in_shapes_recv_host(world_size);
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait());
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(in_shapes_recv_host.data(),
                                            in_shape_device,
                                            kSize * world_size,
                                            XPUMemcpyKind::XPU_DEVICE_TO_HOST));
      VLOG(3) << "Dynamic check recv metadata, shape: "
              << paddle::string::join_strings(in_shapes_recv_host, ',');
      for (int out_rank = 0; out_rank < world_size; ++out_rank) {
        CheckShape(out_tensor[out_rank], in_shapes_recv_host[out_rank]);
      }
    }
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_free(in_shape_device));
  }
}

}  // namespace phi::distributed
