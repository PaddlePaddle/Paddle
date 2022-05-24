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
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

enum { kTransposeMKLDNNFP32 = 1, kTransposeMKLDNNINT8 = 2 };

template <typename DeviceContext, typename T>
inline void TransCompute(const int dim, const DeviceContext& dev_ctx,
                         const framework::Tensor& in, framework::Tensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
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

template <typename T, size_t kMaxMovementSize>
class DimsAndPermSimplifier {
 public:
  explicit DimsAndPermSimplifier(const int rank, const int elem_size,
                                 const std::vector<int32_t>& perm,
                                 std::vector<size_t>* in_dims, const T* src,
                                 T* dst) {
    perm_.resize(rank);
    dims_.resize(rank);
    size_t simplified_movement_size =
        GetMovementSize(elem_size, rank, *in_dims, perm, src, dst);

    (*in_dims)[rank - 1] /= (simplified_movement_size / elem_size);
    Simplifyperm(rank, *in_dims, perm);

    movement_size_ = GetMovementSize(simplified_movement_size, rank_, dims_,
                                     perm_, src, dst);
    dims_[rank_ - 1] /= (movement_size_ / simplified_movement_size);
  }

  size_t GetRank() const { return rank_; }
  size_t GetMovementSize() const { return movement_size_; }
  bool IsSequential() const { return sequential_flag_; }

  std::vector<int> GetPerm() const { return perm_; }
  std::vector<size_t> GetDims() const { return dims_; }

 private:
  size_t rank_{1};
  size_t movement_size_;
  bool sequential_flag_{false};
  std::vector<int> perm_;
  std::vector<size_t> dims_;

  void Simplifyperm(const size_t rank, const std::vector<size_t>& in_dims,
                    const std::vector<int32_t>& perm) {
    size_t combined_dims[phi::DDim::kMaxRank];
    int valid_map[phi::DDim::kMaxRank];

    // Merge consecutive dims to the fist one of this these dims,
    // and leave the origin dim value to be 1. Example below :
    // perm: [2, 3, 0, 1], origin_dims : [4, 8, 2, 5]
    // new_dims: [4, 8, 2, 5] -> [32, 1, 10, 1]
    size_t start_perm_idx = 0;
    while (start_perm_idx < rank) {
      const size_t start_dim_idx = perm[start_perm_idx];
      combined_dims[start_dim_idx] = in_dims[start_dim_idx];
      size_t end_perm_idx = start_perm_idx + 1;

      while (end_perm_idx < rank &&
             perm[end_perm_idx] == perm[end_perm_idx - 1] + 1) {
        const size_t end_dim_idx = perm[end_perm_idx];
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
    size_t valid_dim_idx = 0;
    bool sequential_flag = false;
    for (size_t i = 0; i < rank; ++i) {
      const int src_dim = combined_dims[i];
      if (src_dim == 1) {
        valid_map[i] = -1;
      } else {
        sequential_flag = true;
        valid_map[i] = valid_dim_idx;
        dims_[valid_dim_idx] = src_dim;
        valid_dim_idx += 1;
      }
    }

    if (valid_dim_idx == 0) {
      dims_[0] = 1;
      perm_[0] = 0;
      return;
    } else if (valid_dim_idx == 1) {
      sequential_flag_ = true;
    }

    // Acquire simplified perm with help of combined dims
    // and original perm, finally simplified perm is [1, 0]
    size_t perm_idx = 0;
    for (size_t i = 0; i < rank; ++i) {
      const int mapped = valid_map[perm[i]];
      if (mapped >= 0) {
        perm_[perm_idx] = mapped;
        perm_idx += 1;
      }
    }
    rank_ = valid_dim_idx;
  }

  size_t GetMovementSize(const size_t elem_size, const size_t rank,
                         const std::vector<size_t>& in_dims,
                         const std::vector<int>& perm, const T* src, T* dst) {
    static_assert(kMaxMovementSize > 0 &&
                      (kMaxMovementSize & (kMaxMovementSize - 1)) == 0,
                  "The kMaxMovementSize shall be power of 2.");

    if (perm[rank - 1] == rank - 1) {
      const size_t last_dim_size = in_dims[rank - 1] * elem_size;
      auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
      auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);

      for (size_t size = kMaxMovementSize; size > elem_size; size /= 2) {
        if (last_dim_size % size == 0 && src_ptr % size == 0 &&
            dst_ptr % size == 0) {
          return size;
        }
      }
    }
    return elem_size;
  }
};

}  // namespace operators
}  // namespace paddle
