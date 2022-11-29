/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

enum { kTransposeMKLDNNFP32 = 1, kTransposeMKLDNNINT8 = 2 };

template <typename DeviceContext, typename T>
inline void TransCompute(const int dim,
                         const DeviceContext& dev_ctx,
                         const phi::DenseTensor& in,
                         phi::DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 0:
      phi::Copy<DeviceContext>(dev_ctx, in, dev_ctx.GetPlace(), false, out);
      break;
    case 1:
      phi::funcs::Transpose<DeviceContext, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      phi::funcs::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      phi::funcs::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      phi::funcs::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      phi::funcs::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      phi::funcs::Transpose<DeviceContext, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      phi::funcs::TransposeNormal<DeviceContext, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

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

#define GETTILESIZE(LEN, ALIGN) ((LEN + (ALIGN - 1)) & ~(ALIGN - 1)) / ALIGN

// Simplify the input dims and permute dims if possible.
template <typename T>
struct DimsSimplifier {
 public:
  explicit DimsSimplifier(const int rank,
                          const int64_t numel,
                          const std::vector<int32_t>& perm,
                          const std::vector<int64_t>& dims)
      : perm_(rank), src_dims_(rank), count_(numel) {
    SimplifyPermAndDims(rank, dims, perm);
    perm_.resize(rank_);
    src_dims_.resize(rank_);
    dst_dims_.resize(rank_);
    if (!is_seq_perm_) {
      for (auto i = 0; i < rank_; ++i) {
        dst_dims_[i] = src_dims_[perm_[i]];
      }
    } else {
      dst_dims_[0] = numel;
      src_dims_[0] = numel;
    }
  }

  ~DimsSimplifier() = default;

  const int& GetRank() const { return rank_; }
  const int64_t& GetCount() const { return count_; }
  const std::vector<int>& GetPerm() const { return perm_; }
  const std::vector<int64_t>& GetSrcDims() const { return src_dims_; }
  const std::vector<int64_t>& GetDstDims() const { return dst_dims_; }

 private:
  int rank_{1};
  int64_t count_{0};
  bool is_seq_perm_{true};
  std::vector<int> perm_;
  std::vector<int64_t> src_dims_;
  std::vector<int64_t> dst_dims_;

  void SimplifyPermAndDims(const int rank,
                           const std::vector<int64_t>& in_dims,
                           const std::vector<int32_t>& perm) {
    int start_perm_idx = 0;
    int valid_dim_idx = 0;
    int valid_map[phi::DDim::kMaxRank];
    int64_t combined_dims[phi::DDim::kMaxRank];

    // Merge consecutive dims to the fist one dim and
    // leave original dim to be 1. Example below :
    // perm: [2, 3, 0, 1], origin_dims : [4, 8, 2, 5]
    // new_dims: [4, 8, 2, 5] -> [32, 1, 10, 1]
    while (start_perm_idx < rank) {
      const int start_dim_idx = perm[start_perm_idx];
      combined_dims[start_dim_idx] = in_dims[start_dim_idx];
      int end_perm_idx = start_perm_idx + 1;

      while (end_perm_idx < rank &&
             perm[end_perm_idx] == perm[end_perm_idx - 1] + 1) {
        const int end_dim_idx = perm[end_perm_idx];
        combined_dims[start_dim_idx] *= in_dims[end_dim_idx];
        combined_dims[end_dim_idx] = 1;
        end_perm_idx += 1;
      }
      start_perm_idx = end_perm_idx;
    }

    // Reorder combined dims and marked useless dim as -1.
    // for example, if combined dims is [32, 1, 10, 1],
    // valid_map is [0, -1, 1, -1] and generate simplified
    // dims as [32, 10]
    for (auto i = 0; i < rank; ++i) {
      const int dim_val = combined_dims[i];
      if (dim_val == 1) {
        valid_map[i] = -1;
      } else {
        valid_map[i] = valid_dim_idx;
        src_dims_[valid_dim_idx] = dim_val;
        valid_dim_idx += 1;
      }
    }

    if (valid_dim_idx == 0) {
      src_dims_[0] = 1;
      perm_[0] = 0;
      return;
    }

    // Acquire simplified perm with help of combined dims
    // and original perm, finally simplified perm is [1, 0]
    int perm_idx = 0;
    for (auto i = 0; i < rank; ++i) {
      const int mapped = valid_map[perm[i]];
      if (mapped >= 0) {
        perm_[perm_idx] = mapped;
        is_seq_perm_ &= (mapped == perm_idx);
        perm_idx += 1;
      }
    }
    rank_ = is_seq_perm_ ? 1 : valid_dim_idx;
  }
};

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
      if ((rank == 2 && perm[1] == 0 && perm[0] == 1) ||
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

}  // namespace operators
}  // namespace paddle
