// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#ifdef __NVCC__
#include "cub/cub.cuh"  // NOLINT
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {
namespace detail {

template <typename T>
struct IdentityFunctor {  // for reduce_sum / reduce_max / reduce_min
  HOSTDEVICE explicit inline IdentityFunctor() {}

  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

template <typename T>
struct DivideFunctor {  // for reduce_mean
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

static inline int GetLastPow2(int n) {
  n = log2(n);
  n = max(0, n);
  return std::pow(2, n);
}

static inline std::vector<int> GetStrides(const std::vector<int>& dims) {
  int n = static_cast<int>(dims.size());
  if (n == 0) return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

static inline std::vector<int> GetStrides(const std::vector<int>& dims,
                                          const std::vector<int>& idx) {
  int n = static_cast<int>(idx.size());
  if (n == 0) return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[idx[i + 1]];
  }
  return strides;
}

#ifdef __HIPCC__
constexpr int kMaxBlockDim = 256;
#else
constexpr int kMaxBlockDim = 512;
#endif

static inline int GetDesiredBlockDim(int block_dim) {
  return block_dim >= kMaxBlockDim
             ? kMaxBlockDim
             : (1 << static_cast<int>(std::log2(block_dim)));
}

static inline void CheckReduceRankIsValid(int reduce_rank, int rank) {
  if (rank % 2 == 0) {
    PADDLE_ENFORCE_EQ(reduce_rank, rank / 2,
                      platform::errors::InvalidArgument(
                          "ReduceOp: invalid reduce rank. When rank = %d, "
                          "reduce_rank must be %d, but got %d.",
                          rank, rank / 2, reduce_rank));
  } else {
    auto lower_rank = (rank - 1) / 2;
    auto upper_rank = (rank + 1) / 2;
    PADDLE_ENFORCE_EQ(
        reduce_rank == lower_rank || reduce_rank == upper_rank, true,
        platform::errors::InvalidArgument(
            "ReduceOp: invalid reduce rank. When rank = %d, reduce_rank "
            "must be %d or %d, but got %d.",
            rank, lower_rank, upper_rank, reduce_rank));
  }
}

template <typename T, size_t ElementCount>
struct Array {
 public:
  HOSTDEVICE inline Array() {}

  HOSTDEVICE inline T& operator[](size_t index) { return data_[index]; }

  HOSTDEVICE inline const T& operator[](size_t index) const {
    return data_[index];
  }

  HOSTDEVICE constexpr inline size_t size() const { return ElementCount; }

  template <typename VectorLikeType>
  static inline Array<T, ElementCount> From(const VectorLikeType& vec) {
    PADDLE_ENFORCE_EQ(vec.size(), ElementCount,
                      platform::errors::InvalidArgument(
                          "Cub reduce Array: size not match. Received "
                          "vec.size() %d !=  ElementCount %d.",
                          vec.size(), ElementCount));
    size_t n = static_cast<size_t>(vec.size());
    Array<T, ElementCount> ret;
    for (size_t i = 0; i < n; ++i) ret[i] = vec[i];
    return ret;
  }

 private:
  T data_[ElementCount];
};

}  // namespace detail
// reduce config

enum ReduceType {
  kReduceAll = 0x00,
  kReduceLastDim = 0x01,
  kReduceFirstDim = 0x02,
  kReduceCommon = 0x03,
};

struct ReduceConfig {
  ReduceConfig(std::vector<int> origin_reduce_dims, std::vector<int> x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(x_dim) {}
  // update x_dim and reduce_dim
  void updateReduceDim() {
    std::set<int> reduce_set;
    std::set<int> x_set;
    for (auto e : reduce_dims_origin) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      reduce_set.insert(pos);
    }
    std::vector<int> reduce_dims(reduce_set.begin(), reduce_set.end());
    std::sort(reduce_dims.begin(), reduce_dims.end());
    if (reduce_dims.size() > 1) {
      for (int i = reduce_dims.size() - 2; i >= 0; i--) {
        int axis = reduce_dims[i + 1];
        if (reduce_dims[i + 1] - reduce_dims[i] == 1) {
          reduce_set.erase(reduce_dims[i + 1]);
        }
      }
    }
    reduce_dims.clear();
    reduce_dims.assign(reduce_set.begin(), reduce_set.end());

    std::vector<int> new_x_dim, new_reduce_dims;
    int is_reduced = 0;
    for (auto e : reduce_dims) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      is_reduced |= 1 << e;
    }

    for (int i = 0; i < x_dim.size(); i++) {
      if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
        new_x_dim.push_back(x_dim[i]);
        if ((is_reduced >> i) & 1)
          new_reduce_dims.push_back(new_x_dim.size() - 1);
      } else {
        new_x_dim[new_x_dim.size() - 1] *= x_dim[i];
      }
    }

    x_dim = new_x_dim;
    reduce_dims = new_reduce_dims;
    int x_rank = static_cast<int>(x_dim.size());
    std::set<int> left_set;
    for (int i = 0; i < x_rank; ++i) left_set.insert(i);

    for (auto e : reduce_dims) {
      left_set.erase(e);
      reduce_dim.push_back(e);
    }
    left_dim.assign(left_set.begin(), left_set.end());
  }

  void UpdateStrides() {
    x_strides = detail::GetStrides(x_dim);
    reduce_strides = detail::GetStrides(x_dim, reduce_dim);
    left_strides = detail::GetStrides(x_dim, left_dim);
    reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];
    left_num = 1;
    if (left_dim.size()) left_num = left_strides[0] * x_dim[left_dim[0]];
  }

  int getReduceType() {
    int rank = x_dim.size();
    int reduce_rank = reduce_dim.size();
    if (rank == reduce_rank) {
      reduce_type = static_cast<int>(ReduceType::kReduceAll);
    } else if (rank == 2 && reduce_rank == 1 && reduce_dim[0] == 1) {
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);
    } else if (rank == 2 && reduce_rank == 1 && reduce_dim[0] == 0) {
      reduce_type = static_cast<int>(ReduceType::kReduceFirstDim);
    } else {
      reduce_type = static_cast<int>(ReduceType::kReduceCommon);
    }
  }
  void setBlockDim() {
    int block_num = detail::GetDesiredBlockDim(reduce_num);
    // init
    reduce_ny = false;
    dim3 block(block_num, 1);
    dim3 grid(left_num, 1);
    if (reduce_type == ReduceType::kReduceFirstDim) {
      int device_id = platform::GetCurrentDeviceId();
      int max_mp = platform::GetCUDAMultiProcessors(device_id);
      int max_threads_per_mp =
          platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
      int max_threads = max_threads_per_mp * max_mp;
      int reduce_nx = x_dim[1];
      int reduce_ny = x_dim[0];
      // init
      block_size_reduce_ny =
          detail::GetLastPow2(reduce_ny / (max_threads / reduce_nx));
      if (max_threads / reduce_nx >= 2 && reduce_ny > 512 &&
          block_size_reduce_ny > 1) {
        reduce_ny = true;
        block.x = 32;
        block.y = 1;
        grid.x = (reduce_nx + block.x - 1) / block.x;
        grid.y = (reduce_ny + block_size_reduce_ny - 1) / block_size_reduce_ny;
      } else {
        block.x = 32;
        block.y = 1;
        block_size_reduce_ny = reduce_ny;
        grid.x = (reduce_nx + block.x - 1) / block.x;
        grid.y = 1;
      }
    }
    block_ = block;
    grid_ = grid;
  }

  bool should_reduce_ny() { return reduce_ny; }

  std::vector<int> reduce_dims_origin;
  std::vector<int> reduce_dim;
  std::vector<int> x_dim;
  std::vector<int> left_dim;
  std::vector<int> x_strides;
  std::vector<int> left_strides;
  std::vector<int> reduce_strides;
  int reduce_type;
  int reduce_num;
  int left_num;
  int block_size_reduce_ny;
  bool reduce_ny;
  dim3 block_;
  dim3 grid_;
};

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim>
__device__ void ReduceLastDim(const Tx* x, Ty* y, ReduceOp reducer,
                              TransformOp transformer, Ty init,
                              int reduce_num) {
  __shared__ typename cub::BlockReduce<Ty, BlockDim>::TempStorage temp_storage;
  int idx_x = blockIdx.x * reduce_num;
  int idx_y = threadIdx.x;
  Ty reduce_var = init;
  for (int idx_y = threadIdx.x; idx_y < reduce_num; idx_y += BlockDim)
    reduce_var =
        reducer(reduce_var, static_cast<Ty>(transformer(x[idx_x + idx_y])));
  __syncthreads();

  reduce_var =
      cub::BlockReduce<Ty, BlockDim>(temp_storage).Reduce(reduce_var, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = reduce_var;
  }
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
__device__ void ReduceFirstDim(const Tx* x, Ty* y, ReduceOp reducer,
                               TransformOp transformer, Ty init, int nx, int ny,
                               int block_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * block_size;
  Ty temp = static_cast<Ty>(0.0f);
  if (idx > nx) return;
  for (int iy = 0; iy < block_size && idy + iy < ny; iy++) {
    int id = (idy + iy) * nx + idx;
    Ty tp = static_cast<Ty>(x[id]);
    temp += tp;
  }
  y[idx + blockIdx.y * nx] = static_cast<Ty>(transformer(temp));
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim, int Rank, int ReduceRank>
__device__ void ReduceAny(const Tx* x, Ty* y, ReduceOp reducer,
                          TransformOp transformer, Ty init, int reduce_num,
                          detail::Array<int, Rank> x_strides,
                          detail::Array<int, ReduceRank> reduce_dim,
                          detail::Array<int, ReduceRank> reduce_strides,
                          detail::Array<int, Rank - ReduceRank> left_dim,
                          detail::Array<int, Rank - ReduceRank> left_strides) {
  __shared__ typename cub::BlockReduce<Ty, BlockDim>::TempStorage temp_storage;
  int sub_index[Rank];
  int left_idx = blockIdx.x;
  for (int i = 0; i < Rank - ReduceRank; ++i) {
    sub_index[left_dim[i]] = left_idx / left_strides[i];
    left_idx %= left_strides[i];
  }

  int reduce_idx = threadIdx.x;
  for (int j = 0; j < ReduceRank; ++j) {
    sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
    reduce_idx %= reduce_strides[j];
  }

  int idx_x = 0;
  for (int k = 0; k < Rank; ++k) idx_x += (sub_index[k] * x_strides[k]);
  Ty reduce_var = static_cast<Ty>(transformer(x[idx_x]));

  for (int i = threadIdx.x + BlockDim; i < reduce_num; i += BlockDim) {
    int reduce_idx = i;
    for (int j = 0; j < ReduceRank; ++j) {
      sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
      reduce_idx %= reduce_strides[j];
    }

    int idx_x = 0;
    for (int k = 0; k < Rank; ++k) idx_x += (sub_index[k] * x_strides[k]);
    reduce_var = static_cast<Ty>(
        reducer(reduce_var, static_cast<Ty>(transformer(x[idx_x]))));
  }
  __syncthreads();

  reduce_var =
      cub::BlockReduce<Ty, BlockDim>(temp_storage).Reduce(reduce_var, reducer);

  if (threadIdx.x == 0) {
    y[blockIdx.x] = reduce_var;
  }
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim, int Rank, int ReduceRank>
__device__ void ReduceDevice(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer, Ty init,
    int reduce_num, int left_num, int blocking_size, int reduce_type,
    detail::Array<int, Rank> x_strides,
    detail::Array<int, ReduceRank> reduce_dim,
    detail::Array<int, ReduceRank> reduce_strides,
    detail::Array<int, Rank - ReduceRank> left_dim,
    detail::Array<int, Rank - ReduceRank> left_strides) {
  if (reduce_type == ReduceType::kReduceLastDim) {
    ReduceLastDim<Tx, Ty, ReduceOp, TransformOp, BlockDim>(
        x, y, reducer, transformer, init, reduce_num);
  } else if (reduce_type == ReduceType::kReduceFirstDim) {
    ReduceFirstDim<Tx, Ty, ReduceOp, TransformOp>(
        x, y, reducer, transformer, init, reduce_num, left_num, blocking_size);
  } else {
    ReduceAny<Tx, Ty, ReduceOp, TransformOp, BlockDim, Rank, ReduceRank>(
        x, y, reducer, transformer, init, reduce_num, x_strides, reduce_dim,
        reduce_strides, left_dim, left_strides);
  }
}

template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp,
          int BlockDim, int Rank, int ReduceRank>
__global__ void ReduceKernel_m(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer, Ty init,
    int reduce_num, int left_num, int block_size, int reduce_type,
    detail::Array<int, Rank> x_strides,
    detail::Array<int, ReduceRank> reduce_dim,
    detail::Array<int, ReduceRank> reduce_strides,
    detail::Array<int, Rank - ReduceRank> left_dim,
    detail::Array<int, Rank - ReduceRank> left_strides) {
  ReduceDevice<Tx, Ty, ReduceOp, TransformOp, BlockDim, Rank, ReduceRank>(
      x, y, reducer, transformer, init, reduce_num, left_num, block_size,
      reduce_type, x_strides, reduce_dim, reduce_strides, left_dim,
      left_strides);
}

template <typename Tx, typename Ty, int BlockDim, typename ReduceOp,
          typename TransformOp>
static void launchReduceKernel(const Tx* x_data, Ty* y_data,
                               const platform::Place& place,
                               const ReduceOp& reducer,
                               const TransformOp& transformer, const Ty& init,
                               gpuStream_t stream, ReduceConfig config) {
  int reduce_rank = config.reduce_strides.size();
  int rank = config.x_strides.size();

#define CUB_RANK_CASE(i, ...)             \
  case i: {                               \
    constexpr auto kRank = i;             \
    switch (reduce_rank) { __VA_ARGS__; } \
  } break

#define CUB_REDUCE_RANK_CASE(i, ...)                                         \
  case i: {                                                                  \
    constexpr auto kReduceRank = i;                                          \
    ReduceKernel_m<Tx, Ty, ReduceOp, TransformOp, BlockDim, kRank,           \
                   kReduceRank><<<config.grid_, config.block_, 0, stream>>>( \
        x_data, y_data, reducer, transformer, init, config.reduce_num,       \
        config.left_num, config.block_size_reduce_ny, config.reduce_type,    \
        detail::Array<int, kRank>::From(config.x_strides),                   \
        detail::Array<int, kReduceRank>::From(config.reduce_dim),            \
        detail::Array<int, kReduceRank>::From(config.reduce_strides),        \
        detail::Array<int, kRank - kReduceRank>::From(config.left_dim),      \
        detail::Array<int, kRank - kReduceRank>::From(config.left_strides)); \
  } break

  // launch CUB::Reduce
  if (config.reduce_type == static_cast<int>(ReduceType::kReduceAll)) {
    cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
        x_data, transformer);
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, init, stream);
    framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
        place);
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, init, stream);

    return;
  }
  detail::CheckReduceRankIsValid(reduce_rank, rank);
  switch (rank) {
    CUB_RANK_CASE(2, CUB_REDUCE_RANK_CASE(1););

    CUB_RANK_CASE(3, CUB_REDUCE_RANK_CASE(1); CUB_REDUCE_RANK_CASE(2););

    CUB_RANK_CASE(4, CUB_REDUCE_RANK_CASE(2););

    CUB_RANK_CASE(5, CUB_REDUCE_RANK_CASE(2); CUB_REDUCE_RANK_CASE(3););

    CUB_RANK_CASE(6, CUB_REDUCE_RANK_CASE(3););

    CUB_RANK_CASE(7, CUB_REDUCE_RANK_CASE(3); CUB_REDUCE_RANK_CASE(4););

    CUB_RANK_CASE(8, CUB_REDUCE_RANK_CASE(4););

    CUB_RANK_CASE(9, CUB_REDUCE_RANK_CASE(4); CUB_REDUCE_RANK_CASE(5););
  }

  if (config.should_reduce_ny()) {
    switch (rank) { CUB_RANK_CASE(2, CUB_REDUCE_RANK_CASE(1);); }
  }

#undef CUB_REDUCE_RANK_CASE
#undef CUB_RANK_CASE
}
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
void TensorReduce(const framework::Tensor& x, framework::Tensor* y,
                  std::vector<int> origin_reduce_dims, const Ty& init,
                  const ReduceOp& reducer, const TransformOp& transformer,
                  gpuStream_t stream) {
  auto x_dim = framework::vectorize<int>(x.dims());
  auto config = ReduceConfig(origin_reduce_dims, x_dim);
  // update Reduce_dim and x_dim
  config.updateReduceDim();
  config.UpdateStrides();
  // malloc
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>(x.place());
  // use copy
  if (config.reduce_num == 1) {
    auto out_dims = y->dims();
    framework::TensorCopy(x, y->place(), y);
    y->Resize(out_dims);
    return;
  }
  // set BlockDim for block and grid
  config.setBlockDim();
// launchReduceKernel
#define CUB_BLOCK_DIM_CASE(block_dim)                                  \
  case block_dim: {                                                    \
    constexpr auto kBlockDim = block_dim;                              \
    launchReduceKernel<Tx, Ty, block_dim, ReduceOp, TransformOp>(      \
        x_data, y_data, x.place(), reducer, transformer, init, stream, \
        config);                                                       \
  } break

  switch (detail::GetDesiredBlockDim(config.reduce_num)) {
    CUB_BLOCK_DIM_CASE(512);
    CUB_BLOCK_DIM_CASE(256);
    CUB_BLOCK_DIM_CASE(128);
    CUB_BLOCK_DIM_CASE(64);
    CUB_BLOCK_DIM_CASE(32);
    CUB_BLOCK_DIM_CASE(16);
    CUB_BLOCK_DIM_CASE(8);
    CUB_BLOCK_DIM_CASE(4);
    CUB_BLOCK_DIM_CASE(2);
  }
}

template <typename Tx, typename ReduceOp, typename TransformOp>
struct TensorReduceFunctor {
  const framework::Tensor& x;
  framework::Tensor* y;
  std::vector<int> origin_reduce_dims;
  const double& init;
  const ReduceOp& reducer;
  const TransformOp& transformer;
  gpuStream_t stream;
  TensorReduceFunctor(const framework::Tensor& x, framework::Tensor* y,
                      std::vector<int> origin_reduce_dims, const double& init,
                      const ReduceOp& reducer, const TransformOp& transformer,
                      gpuStream_t stream)
      : x(x),
        y(y),
        origin_reduce_dims(origin_reduce_dims),
        init(init),
        reducer(reducer),
        transformer(transformer),
        stream(stream) {}

  template <typename Ty>

  void apply() const {
    const Ty& init_cast = static_cast<Ty>(init);
    TensorReduce<Tx, Ty, ReduceOp, TransformOp>(
        x, y, origin_reduce_dims, init_cast, reducer, transformer, stream);
  }
};

}  // namespace operators
}  // namespace paddle
