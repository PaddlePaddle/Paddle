/*
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
*/
#include "./utils.h"
#include "paddle/common/array.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

template <int MAX_NUM_EXPERTS>
struct __align__(16) expert_base_offset {
  int data[MAX_NUM_EXPERTS];
};

template <typename ZipT, typename UnzipT, int VecSize>
__global__ void tokens_zip_unique_add_kernel(
    ZipT *__restrict__ zipped,
    const UnzipT *__restrict__ unzipped,
    const int64_t *__restrict__ index_unzipped,
    const int64_t unzipped_rows,
    const int hidden_size) {
  for (int64_t unzipped_row = blockIdx.x; unzipped_row < unzipped_rows;
       unzipped_row += gridDim.x) {
    auto *zipped_ptr = zipped + index_unzipped[unzipped_row] * hidden_size;
    const auto *unzipped_ptr = unzipped + unzipped_row * hidden_size;
    for (int i = threadIdx.x * VecSize; i < hidden_size;
         i += blockDim.x * VecSize) {
      phi::AlignedVector<ZipT, VecSize> zipped_tmp;
      phi::AlignedVector<UnzipT, VecSize> unzipped_tmp;
      phi::Load(zipped_ptr + i, &zipped_tmp);
      phi::Load(unzipped_ptr + i, &unzipped_tmp);
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        zipped_tmp[j] += static_cast<ZipT>(unzipped_tmp[j]);
      }
      phi::Store(zipped_tmp, zipped_ptr + i);
    }
  }
}

std::vector<paddle::Tensor> tokens_zip_unique_add(
    const paddle::Tensor &zipped_origin,
    const paddle::Tensor &unzipped,
    const paddle::Tensor &index_unzipped,
    int64_t zipped_rows) {
  auto zipped_shape = zipped_origin.shape();
  auto unzipped_shape = unzipped.shape();
  PD_CHECK(zipped_shape.size() == 2);
  PD_CHECK(unzipped_shape.size() == 2);
  PD_CHECK(zipped_shape[1] == unzipped_shape[1]);

  auto hidden_size = zipped_shape[1];

  auto out_dtype = zipped_origin.dtype();
  auto in_dtype = unzipped.dtype();
  auto place = zipped_origin.place();

  paddle::Tensor zipped;
  if (zipped_shape[0] == 0) {
    zipped = paddle::zeros({zipped_rows, hidden_size}, out_dtype, place);
  } else {
    PD_CHECK(zipped_shape[0] == zipped_rows);
    zipped = zipped_origin;
  }

  auto index_shape = index_unzipped.shape();
  PD_CHECK(index_shape.size() == 1);
  auto unzipped_rows = index_shape[0];
  PD_CHECK(unzipped_rows <= zipped_rows);
  PD_CHECK(unzipped_rows <= unzipped_shape[0]);

  constexpr int kVecSize = 4;
  PD_CHECK(hidden_size % kVecSize == 0);

  int block = 1024;
  int grid = LimitGridDim(unzipped_rows);

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD(__ZipT, __UnzipT)       \
  do {                                                       \
    tokens_zip_unique_add_kernel<__ZipT, __UnzipT, kVecSize> \
        <<<grid, block, 0, unzipped.stream()>>>(             \
            zipped.data<__ZipT>(),                           \
            unzipped.data<__UnzipT>(),                       \
            index_unzipped.data<int64_t>(),                  \
            unzipped_rows,                                   \
            hidden_size);                                    \
  } while (0)

  if (grid > 0) {
    if (out_dtype == paddle::DataType::FLOAT32 &&
        in_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::BFLOAT16 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(phi::bfloat16, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::FLOAT32 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, float);
    } else {
      PD_THROW("Unsupported data type");
    }
  }

  return {zipped};
}

PD_BUILD_OP(tokens_zip_unique_add)
    .Inputs({"x_zipped", "x_unzipped", "idx_unzipped"})
    .Outputs({"y_zipped"})
    .Attrs({"zipped_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_unique_add));
