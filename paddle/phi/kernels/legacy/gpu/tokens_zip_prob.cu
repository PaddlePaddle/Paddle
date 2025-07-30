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

namespace phi {

template <int MAX_NUM_EXPERTS>
struct __align__(16) expert_base_offset {
  int data[MAX_NUM_EXPERTS];
};

template <typename T>
struct UnzippedProbInfo {
  const T *__restrict__ data;
  int64_t offset;
};

template <typename T, int MAX_NUM_EXPERTS_C>
__global__ void tokens_zip_prob_kernel(
    phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ dispatched_indices,
    T *zipped_probs,
    int64_t zipped_rows,
    int topk,
    int num_expert) {
  int64_t idx = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t limit = zipped_rows * topk;
  while (idx < limit) {
    auto zipped_row = idx / topk;
    auto topk_idx = idx % topk;
    auto expert_id = dispatched_indices[idx];
    T value = static_cast<T>(0);
    if (expert_id >= 0) {
      auto unzipped_row =
          zipped_expertwise_rowmap[zipped_row * num_expert + expert_id];
      if (unzipped_row >= 0) {
        unzipped_row -= unzipped_probs[expert_id].offset;
        value = unzipped_probs[expert_id].data[unzipped_row];
      }
    }
    zipped_probs[idx] = value;
    idx += stride;
  }
}

template <typename T>
std::vector<paddle::Tensor> tokens_zip_prob_impl(
    const std::vector<paddle::Tensor> &unzipped_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const paddle::Tensor &dispatched_indices,
    paddle::DataType dtype) {
  auto zipped_expertwise_rowmap_shape = zipped_expertwise_rowmap.shape();
  auto dispatched_indices_shape = dispatched_indices.shape();
  PD_CHECK(zipped_expertwise_rowmap_shape.size() == 2);
  PD_CHECK(dispatched_indices_shape.size() == 2);
  PD_CHECK(zipped_expertwise_rowmap_shape[0] == dispatched_indices_shape[0]);

  int64_t zipped_rows = zipped_expertwise_rowmap_shape[0];
  int num_expert = zipped_expertwise_rowmap_shape[1];
  int topk = dispatched_indices_shape[1];
  PD_CHECK(unzipped_probs.size() == num_expert);

  auto zipped_probs =
      paddle::empty({zipped_rows, topk}, dtype, unzipped_probs[0].place());

  PD_SWITCH_NUM_EXPERTS(
      num_expert, ([&] {
        phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs_info;
        int64_t offset = 0;
        for (int i = 0; i < num_expert; ++i) {
          auto shape = unzipped_probs[i].shape();
          PD_CHECK(shape.size() == 1);
          unzipped_probs_info[i].data = unzipped_probs[i].data<T>();
          unzipped_probs_info[i].offset = offset;
          offset += shape[0];
        }

        int thread = 1024;
        int grid = LimitGridDim((zipped_rows * topk + thread - 1) / thread);

        if (grid > 0) {
          tokens_zip_prob_kernel<T, MAX_NUM_EXPERTS_C>
              <<<grid, thread, 0, zipped_probs.stream()>>>(
                  unzipped_probs_info,
                  zipped_expertwise_rowmap.data<int>(),
                  dispatched_indices.data<int>(),
                  zipped_probs.data<T>(),
                  zipped_rows,
                  topk,
                  num_expert);
        }
      }));
  return {zipped_probs};
}

std::vector<paddle::Tensor> TokensZipProbKernel(
    const std::vector<paddle::Tensor> &unzipped_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const paddle::Tensor &dispatched_indices) {
  PD_CHECK(zipped_expertwise_rowmap.dtype() == paddle::DataType::INT32);
  PD_CHECK(dispatched_indices.dtype() == paddle::DataType::INT32);

  auto dtype = unzipped_probs[0].dtype();
  if (dtype == paddle::DataType::FLOAT32) {
    return tokens_zip_prob_impl<float>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return tokens_zip_prob_impl<phi::bfloat16>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else {
    PD_THROW("Unsupported data type: %s", dtype);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(tokens_zip_prob,
                   GPU,
                   ALL_LAYOUT,
                   phi::TokensZipProbKernel,
                   phi::bfloat16,
                   float,
                   double) {}
