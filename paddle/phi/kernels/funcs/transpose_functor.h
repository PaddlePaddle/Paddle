/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

enum { kTransposeMKLDNNFP32 = 1, kTransposeMKLDNNINT8 = 2 };

enum PermuteType {
  kCopy = 1,
  kSwapTranspose = 2,
  kGeneralTranspose = 3,
  kVecPermute = 4,
  kGeneralPermute = 5
};

constexpr int kBlockRows = 16;
constexpr int kTileSize = 32;
constexpr int kShareCol = (kTileSize + 1);

#define GETTILESIZE(LEN_, ALIGN_) \
  ((LEN_ + (ALIGN_ - 1)) & ~(ALIGN_ - 1)) / ALIGN_

template <typename T>
struct PermTypeClassifier {
 public:
  explicit PermTypeClassifier(const int sm_count,
                              const int rank,
                              const std::vector<int32_t>& perm,
                              const std::vector<int64_t>& dims,
                              const T* src,
                              T* dst) {
    if (rank == 1) {
      type_ = PermuteType::kCopy;
    } else {
      // Limitation of the setting in one dimension of cuda grid.
      constexpr int64_t dim_limitation = 65536;
      const int dst_vec_size = phi::GetVectorizedSize<T>(dst);

      // While the last dim is fixed, there is chance for vectorized IO.
      const int last_idx = rank - 1;
      if (perm[last_idx] == last_idx) {
        type_ = PermuteType::kVecPermute;
        vec_size_ = GetDimVecSize(dst_vec_size, dims[last_idx], src, false);
        return;
      }

      // Permute at last 2 dims, namely transpose.
      if ((rank == 2 && perm[1] == 0) ||
          (rank == 3 && perm[2] == 1 && perm[1] == 2)) {
        int64_t channel = rank == 2 ? 1 : dims[0];
        // Currently, transpose kernel cannot cover the case that channel
        // dimension is more than 65536 which is the limitation of dim3 setting.
        // This special case will be covered by extended transpose kernel later.
        if (channel < dim_limitation) {
          type_ = PermuteType::kGeneralTranspose;
          num_rows_tile_ = GETTILESIZE(dims[rank - 2], kTileSize);
          int dim_vec_size = GetDimVecSize(dst_vec_size, dims[last_idx], src);
          int tile_size =
              channel * num_rows_tile_ * GETTILESIZE(dims[last_idx], kTileSize);
          vec_size_ = tile_size < sm_count ? 1 : dim_vec_size;
        } else {
          type_ = PermuteType::kGeneralPermute;
        }
        return;
      }

      // Permute at first dim and third dim.
      if (rank == 3 && perm[2] == 0 && perm[1] == 1) {
        // Currently, transpose kernel cannot cover the case that channel
        // dimension is more than 65536 which is the limitation of dim3 setting.
        // This special case will be covered by extended transpose kernel later.
        if (dims[1] < dim_limitation) {
          type_ = PermuteType::kSwapTranspose;
          num_rows_tile_ = GETTILESIZE(dims[0], kTileSize);

          int dim_vec_size = GetDimVecSize(dst_vec_size, dims[last_idx], src);
          int tile_size =
              dims[1] * num_rows_tile_ * GETTILESIZE(dims[2], kTileSize);
          vec_size_ = tile_size < sm_count ? 1 : dim_vec_size;
        } else {
          type_ = PermuteType::kGeneralPermute;
        }
        return;
      }
      vec_size_ = dst_vec_size;
    }
  }

  ~PermTypeClassifier() = default;

  int GetVecSize() const { return vec_size_; }
  int GetRowsTile() const { return num_rows_tile_; }
  PermuteType GetPermType() const { return type_; }

 private:
  int vec_size_{1};
  int64_t num_rows_tile_{0};
  PermuteType type_{kGeneralPermute};

  // To find if highest common divisor and make it as vec_size.
  int GetDimVecSize(const int dst_vec_size,
                    const int64_t target_dim,
                    const T* src,
                    bool use_share_mem = true) {
    const int vec_size = std::min(dst_vec_size, phi::GetVectorizedSize<T>(src));
    int dim_vec_size = 1;
    for (int size = vec_size; size > 0; size /= 2) {
      if (target_dim % size == 0) {
        dim_vec_size = size;
        break;
      }
    }

    if (use_share_mem) {
      // By bytes limitation of shared_memory.
      return (sizeof(T) > sizeof(float) ? 1 : dim_vec_size);
    } else {
      return dim_vec_size;
    }
  }
};

}  // namespace funcs
}  // namespace phi
