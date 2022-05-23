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
        GetMovementSize(rank, elem_size, *in_dims, perm, src, dst);

    (*in_dims)[rank - 1] /= (simplified_movement_size / elem_size);
    Simplifyperm(rank, *in_dims, perm);

    movement_size_ = GetMovementSize(rank_, simplified_movement_size, dims_,
                                     perm_, src, dst);
    dims_[rank_ - 1] /= (movement_size_ / simplified_movement_size);
  }

  size_t GetRank() const { return rank_; }
  size_t GetMovementSize() const { return movement_size_; }

  std::vector<int> GetPerm() const { return perm_; }
  std::vector<size_t> GetDims() const { return dims_; }

 private:
  size_t rank_;
  size_t movement_size_;
  std::vector<int> perm_;
  std::vector<size_t> dims_;

  void Simplifyperm(const size_t rank, const std::vector<size_t>& in_dims,
                    const std::vector<int32_t>& perm) {
    size_t coalesced_dims[phi::DDim::kMaxRank];
    size_t start_perm_index = 0;

    // Merge consecutive dims for in_dims.
    while (start_perm_index < rank) {
      const size_t start_dim_index = perm[start_perm_index];
      coalesced_dims[start_dim_index] = in_dims[start_dim_index];
      size_t end_perm_index = start_perm_index + 1;

      while (end_perm_index < rank &&
             perm[end_perm_index] == perm[end_perm_index - 1] + 1) {
        const size_t end_dim_index = perm[end_perm_index];
        coalesced_dims[start_dim_index] *= in_dims[end_dim_index];
        coalesced_dims[end_dim_index] = 1;
        end_perm_index += 1;
      }
      start_perm_index = end_perm_index;
    }

    // Merge value `1` dim for perm.
    size_t valid_num_dims = 0;
    int mapping[phi::DDim::kMaxRank];
    for (size_t i = 0; i < rank; ++i) {
      const int src_dim = coalesced_dims[i];
      if (src_dim == 1) {
        mapping[i] = -1;
      } else {
        mapping[i] = valid_num_dims;
        dims_[valid_num_dims] = src_dim;
        valid_num_dims += 1;
      }
    }

    // Acquire simplified perm.
    if (valid_num_dims == 0) {
      rank_ = 1;
      dims_[0] = 1;
      perm_[0] = 0;
    } else {
      size_t perm_index = 0;
      rank_ = valid_num_dims;
      for (size_t i = 0; i < rank; ++i) {
        const int mapped = mapping[perm[i]];
        if (mapped >= 0) {
          perm_[perm_index] = mapped;
          perm_index += 1;
        }
      }
    }
  }

  size_t GetMovementSize(const size_t rank, const size_t elem_size,
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
