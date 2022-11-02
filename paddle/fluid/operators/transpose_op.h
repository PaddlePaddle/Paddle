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
  kVecPermute = 3,
  kGeneralPermute = 5
};

constexpr int kBlockRows = 16;
constexpr int kTileSize = 32;
constexpr int kShareCol = (kTileSize + 1);

#define GETTILESIZE(LEN, ALIGN) ((LEN + (ALIGN - 1)) & ~(ALIGN - 1)) / ALIGN

// Simplify the input dims and permute dims if possible.
template <typename T>
class DimsSimplifier {
 public:
  explicit DimsSimplifier(const size_t rank,
                          const int64_t numel,
                          const std::vector<int32_t>& perm,
                          const std::vector<int64_t>& dims)
      : perm_(rank), src_dims(rank), count_(numel) {
    SimplifyPermAndDims(rank, dims, perm);
    perm_.resize(rank_);
    src_dims.resize(rank_);
    dst_dims.resize(rank_);

    for (auto i = 0; i < rank_; ++i) {
      dst_dims[i] = src_dims[perm_[i]];
    }
  }

  ~DimsSimplifier() = default;

  int GetRank() const { return rank_; }
  int64_t GetCount() const { return count_; }
  std::vector<int> GetPerm() const { return perm_; }
  std::vector<int64_t> GetSrcDims() const { return src_dims; }
  std::vector<int64_t> GetDstDims() const { return dst_dims; }

 private:
  int rank_{1};
  int64_t count_{0};
  std::vector<int> perm_;
  std::vector<int64_t> src_dims;
  std::vector<int64_t> dst_dims;

  void SimplifyPermAndDims(const size_t rank,
                           const std::vector<int64_t>& in_dims,
                           const std::vector<int32_t>& perm) {
    int start_perm_idx = 0;
    int valid_dim_idx = 0;
    int perm_idx = 0;
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
        src_dims[valid_dim_idx] = dim_val;
        valid_dim_idx += 1;
      }
    }

    if (valid_dim_idx == 0) {
      src_dims[0] = 1;
      perm_[0] = 0;
      return;
    }

    // Acquire simplified perm with help of combined dims
    // and original perm, finally simplified perm is [1, 0]
    for (auto i = 0; i < rank; ++i) {
      const int mapped = valid_map[perm[i]];
      if (mapped >= 0) {
        perm_[perm_idx] = mapped;
        perm_idx += 1;
      }
    }
    rank_ = valid_dim_idx;
  }
};

template <typename T>
class PermTypeClassifier {
 public:
  explicit PermTypeClassifier(const int sm_count,
                              const int rank,
                              const std::vector<int32_t>& perm,
                              const std::vector<int64_t>& dims,
                              const T* src,
                              T* dst) {
    if (rank == 1) {
      vec_size_ = 1;
      type_ = PermuteType::kCopy;
    } else {
      int dst_vec_size = phi::GetVectorizedSize<T>(dst);

      // While the last dim is fixed, there is chance for vectorized IO.
      if (perm[rank - 1] == rank - 1) {
        const int dim_vec_size =
            GetDimVecSize(dst_vec_size, dims[rank - 1], src, false);
        if (dim_vec_size > 1) {
          type_ = kVecPermute;
          vec_size_ = dim_vec_size;
          return;
        }
      }

      // Permute at last 2 dims, namely transpose.
      if ((rank == 2 && perm[1] == 0 && perm[0] == 1) ||
          (rank == 3 && perm[2] == 1 && perm[1] == 2)) {
        type_ = PermuteType::kGeneralTranspose;
        num_cols_tile = GETTILESIZE(dims[rank - 1], kTileSize);
        num_rows_tile = GETTILESIZE(dims[rank - 2], kTileSize);

        int dim_vec_size = GetDimVecSize(dst_vec_size, dims[rank - 1], src);
        int tile_size =
            (rank == 2 ? 1 : dims[0]) * num_cols_tile * num_rows_tile;
        vec_size_ = tile_size < sm_count ? 1 : dim_vec_size;
        return;
      }

      // Permute at first dim and third dim.
      if (rank == 3 && perm[2] == 0 && perm[1] == 1) {
        type_ = PermuteType::kSwapTranspose;
        num_cols_tile = GETTILESIZE(dims[2], kTileSize);
        num_rows_tile = GETTILESIZE(dims[0], kTileSize);

        int dim_vec_size = GetDimVecSize(dst_vec_size, dims[rank - 1], src);
        int tile_size = dims[1] * num_cols_tile * num_rows_tile;
        vec_size_ = tile_size < sm_count ? 1 : dim_vec_size;
        return;
      }

      vec_size_ = dst_vec_size;
    }
  }

  PermTypeClassifier(const PermTypeClassifier& obj) {
    type_ = obj.type_;
    vec_size_ = obj.vec_size_;
    num_rows_tile = obj.num_rows_tile;
  }

  ~PermTypeClassifier() = default;
  int GetVecSize() const { return vec_size_; }
  PermuteType GetPermType() const { return type_; }

 public:
  int num_cols_tile{0};
  int num_rows_tile{0};

 private:
  int vec_size_{1};
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

/*
template <typename T, typename IndexT>
class PermuteDispatch {
 public :
  void operator()(const phi::GPUContext& ctx,
                  const PermTypeClassifier<T>& classifier,
                  const std::vector<int64_t>& dims,
                  const std::vector<int32_t>& perm,
                  const IndexT count,
                  const T* src,
                  T* dst) : classifier_(classifier), dims_(dims), perm_(perm),
count_(count), perm_type_(classifier.GetPermType()) { rank_ = dims_.size();
    KernelDispatch(ctx, classifier.GetVecSize(), src, dst);
  }

  void KernelDispatch(const phi::GPUContext& ctx,
                      const int vec_size,
                      const T* src,
                      T* dst) {
#define CALL_DISPATCH_VEC_SIZE(func_name, vec_size_)    \
    case vec_size_: {                                   \
        func_name<T, IndexT, vec_size_>(ctx, src, dst); \
        break;                                          \
    }

    switch (perm_type_) {
      case kSwapTranspose :
      case kGeneralTranspose :
        switch (vec_size) {
          CALL_DISPATCH_VEC_SIZE(LaunchTransposeKernel, 1);
          CALL_DISPATCH_VEC_SIZE(LaunchTransposeKernel, 2);
          CALL_DISPATCH_VEC_SIZE(LaunchTransposeKernel, 4);
        }
        break;
      default :
        switch (vec_size) {
          CALL_DISPATCH_VEC_SIZE(LaunchPermuteDispatch, 1);
          CALL_DISPATCH_VEC_SIZE(LaunchPermuteDispatch, 2);
          CALL_DISPATCH_VEC_SIZE(LaunchPermuteDispatch, 4);
        }
        break;
    }
#define CALL_DISPATCH_VEC_SIZE
  }

  // A Gerneral permute method.
  void LaunchPermuteDispatch(const phi::GPUContext& ctx,
                             const T* src,
                             T* dst) {
#define CALL_PERMUTE_DISPATCH_RANK(rank)                 \
    case rank: {                                         \
      LaunchPermuteKernel<VecSize, rank>(ctx, src, dst); \
      break;                                             \
    }

    switch (rank_) {
      CALL_PERMUTE_DISPATCH_RANK(3);
      CALL_PERMUTE_DISPATCH_RANK(4);
      CALL_PERMUTE_DISPATCH_RANK(5);
      CALL_PERMUTE_DISPATCH_RANK(6);
      CALL_PERMUTE_DISPATCH_RANK(7);
      CALL_PERMUTE_DISPATCH_RANK(8);
      CALL_PERMUTE_DISPATCH_RANK(9);
    }
#undef CALL_PERMUTE_DISPATCH_RANK
  }

  template <int VecSize, int Rank>
  void LaunchPermuteKernel(const phi::GPUContext& ctx, const T* src, T* dst) {
    size_t main_num = count_ / VecSize;
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, main_num);

    if (perm_type_ == PermuteType::kVecPermute) {
      dims_[rank_ - 1] /= VecSize;
      auto params = PermuteParams<Rank, IndexT>(dims_, perm_);

      VectorizedPermuteKernel<T, IndexT, VecSize, Rank>
          <<<config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
              params, main_num, src, dst);
    } else {
      size_t tail_num = count_ - main_num * VecSize;
      size_t main_offset = count_ - tail_num;
      auto params = PermuteParams<Rank, IndexT>(dims_, perm_);

      GeneralPermuteKernel<T, IndexT, VecSize, Rank>
          <<<config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
              params, src, dst, main_num, tail_num, main_offset);
    }
  }


  // A Gerneral transpose method.
  template <int VecSize>
  void LaunchTransposeKernel(const phi::GPUContext& ctx, const T* src, T* dst) {
    constexpr bool kIsSmallByte = sizeof(T) < sizeof(float);
    constexpr int ReadVecSize = (sizeof(T) > sizeof(float);) ? 1 : VecSize;
    const bool IsTrans = perm_type_ == PermuteType::kGeneralTranspose;

    IndexT chs = IsTrans ? ((rank_ == 2) ? 1 : dims_[0]) : dims_[rank_ - 2];
    IndexT rows = IsTrans ? dims_[rank_ - 2] : dims_[0];
    IndexT cols = dims_[rank_ - 1] / VecSize;
    IndexT num_tile_cols = classifier_.num_tile_cols;

    int vec_write = 1;
    bool is_vec_write = kIsSmallByte ? (rows % (sizeof(float) / sizeof(T)) ?
false : true) : false;

    if (is_vec_write) {
      is_vec_write = (chs * num_tile_cols * classifier_.num_tile_rows) >
ctx.GetSMCount(); vec_write = is_vec_write ? sizeof(float) / sizeof(T) : 1;
    }
    const IndexT num_tile_rows = (vec_write == 1)
                                 ? classifier_.num_tile_rows
                                 : GETTILESIZE(rows, (kTileSize * vec_write));
    dim3 blocks(num_tile_cols, num_tile_rows, chs);
    dim3 threads(kTileSize, kBlockRows, 1);

    if (is_write_size) {
      if (IsTrans) {
        BatchTransposeKernel<T, IndexT, true, VecSize>
         <<<blocks, threads, 0, ctx.stream()>>>(
            src,
            dst,
            num_tile_rows - 1,
            num_tile_cols - 1,
            cols,
            rows);
      } else {
        SwapTransposeKernel<T, IndexT, true, ReadVecSize>
          <<<blocks, threads, 0, ctx.stream()>>>(
            src,
            dst,
            num_tile_rows - 1,
            num_tile_cols - 1,
            cols,
            rows,
            chs);
      }
    } else {
      if (IsTrans) {
        BatchTransposeKernel<T, IndexT, false, VecSize>
         <<<blocks, threads, 0, ctx.stream()>>>(
            src,
            dst,
            num_tile_rows - 1,
            num_tile_cols - 1,
            cols,
            rows);
      } else {
        SwapTransposeKernel<T, IndexT, false, ReadVecSize>
          <<<blocks, threads, 0, ctx.stream()>>>(
            src,
            dst,
            num_tile_rows - 1,
            num_tile_cols - 1,
            cols,
            rows,
            chs);
      }
    }
  }

 private :
  int rank_{0};
  std::vector<int64_t> dims_;
  std::vector<int32_t> perm_;
  PermTypeClassifier<T>& classifier_;
  PermuteType perm_type_{kGeneralPermute};
};
*/
