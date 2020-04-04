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

#include <cstdio>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/float16.h"
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<paddle::platform::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t,
                 paddle::platform::float16> {};
}  // namespace cub

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int64_t id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int64_t id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v < value);
  }

  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (v < in.v) || ((v == in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (v > in.v) || ((v == in.v) && (id < in.id));
  }

  T v;
  int64_t id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int beam_size>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* val, int* col,
                                        int idx, int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < val[idx]) {
      Pair<T> tmp(val[idx], col[idx]);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* val, int* col,
                                        int idx, int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < val[idx]) {
      Pair<T> tmp(val[idx], col[idx]);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* src,
                                              bool* firstStep, bool* is_empty,
                                              Pair<T>* max, int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                              length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -static_cast<T>(1)) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* val,
                                              int* col, bool* firstStep,
                                              bool* is_empty, Pair<T>* max,
                                              int dim, const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, val, col, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - *beam) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, val, col, tid, dim, max,
                              length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T>* sh_topk, int* maxid,
                                            Pair<T> topk[], T** topVal,
                                            int64_t** topIds, int* beam, int* k,
                                            const int tid, const int warp) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
        maxid[tid] = tid + BlockSize / 2;
      } else {
        maxid[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
          maxid[tid] = maxid[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = sh_topk[maxid[0]].v;
      **topIds = sh_topk[maxid[0]].id;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }
    // NOTE(zcd): temporary solution
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (platform::CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) ==
          MaxLength)
        break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to sh_topk, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */

template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixTopK(T* output, int output_stride, int64_t* indices,
                             const T* src, int lds, int dim, int k,
                             int grid_dim, int num) {
  __shared__ Pair<T> sh_topk[BlockSize];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;

  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    int top_num = k;
    __shared__ int maxid[BlockSize / 2];
    T* out = output + i * output_stride;
    int64_t* inds = indices + i * k;
    Pair<T> topk[MaxLength];
    int beam = MaxLength;
    Pair<T> max;
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      topk[j].set(-static_cast<T>(INFINITY), -1);
    }
    while (top_num) {
      ThreadGetTopK<T, MaxLength, BlockSize>(
          topk, &beam, k, src + i * lds, &firststep, &is_empty, &max, dim, tid);

      sh_topk[tid] = topk[0];
      BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &out, &inds,
                                           &beam, &top_num, tid, warp);
    }
  }
}

template <typename T, int MaxLength, int BlockSize>
__global__ void AssignGrad(T* x_grad, const int64_t* indices, const T* out_grad,
                           size_t rows, size_t cols, size_t k) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      x_grad[i * cols + j] = 0;
    }
    for (size_t j = 0; j < k; ++j) {
      size_t idx = indices[i * k + j];
      x_grad[i * cols + idx] = out_grad[i * k + j];
    }
  }
}

inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

// Iter for move to next row
struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

// Iter using into a column
struct ColumnIndexIter {
  explicit ColumnIndexIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }

  int num_cols_;
};

__global__ void InitIndex(int64_t* indices, int64_t num_rows,
                          int64_t num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (int64_t j = row_id; j < num_rows; j += gridDim.x) {
    for (int64_t i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

template <typename T>
bool SortTopk(const platform::CUDADeviceContext& ctx,
              const framework::Tensor* input_tensor, const int64_t num_cols,
              const int64_t num_rows, const int k,
              framework::Tensor* out_tensor,
              framework::Tensor* indices_tensor) {
  auto cu_stream = ctx.stream();

  Tensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = framework::make_ddim(dims);
  input_indices.Resize(dim);
  // input_indices.Resize(num_rows*num_cols);
  input_indices.mutable_data<int64_t>(ctx.GetPlace());
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };

  int block_size = ComputeBlockSize(num_cols);

  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize().x;
  // actually, int num_rows < max_grid_size
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  // Init a index array
  InitIndex<<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<int64_t>(), num_rows, num_cols);

  // create iter for counting input
  cub::CountingInputIterator<int64_t> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<int64_t, SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  T* sorted_values_ptr;
  int64_t* sorted_indices_ptr;

  Tensor temp_values;
  Tensor temp_indices;

  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  int64_t* indices = indices_tensor->mutable_data<int64_t>(ctx.GetPlace());

  if (k == num_cols) {
    // Doing a full sort.
    sorted_values_ptr = values;
    sorted_indices_ptr = indices;
  } else {
    temp_values.Resize(dim);
    temp_indices.Resize(dim);
    sorted_values_ptr = temp_values.mutable_data<T>(ctx.GetPlace());
    sorted_indices_ptr = temp_indices.mutable_data<int64_t>(ctx.GetPlace());
  }

  // Get temp storage buffer size, maybe can allocate a fixed buffer to save
  // time.
  auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_storage_bytes, input, sorted_values_ptr,
      input_indices.data<int64_t>(), sorted_indices_ptr, num_cols * num_rows,
      num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
      cu_stream);
  if (err != cudaSuccess) {
    LOG(ERROR)
        << "TopKOP failed as could not launch "
           "cub::DeviceSegmentedRadixSort::SortPairsDescending to calculate "
           "temp_storage_bytes, status: "
        << cudaGetErrorString(err);
    return false;
  }
  Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp_storage.data<uint8_t>(), temp_storage_bytes, input,
      sorted_values_ptr, input_indices.data<int64_t>(), sorted_indices_ptr,
      num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
      0, sizeof(T) * 8, cu_stream);
  if (err != cudaSuccess) {
    LOG(ERROR)
        << "TopKOP failed as could not launch "
           "cub::DeviceSegmentedRadixSort::SortPairsDescending to sort input, "
           "temp_storage_bytes: "
        << temp_storage_bytes << ", status: " << cudaGetErrorString(err);
    return false;
  }
  auto& dev = *ctx.eigen_device();
  if (k < num_cols) {
    // copy sliced data to output.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, k};
    auto e_indices = EigenMatrix<int64_t>::From(*indices_tensor, dim);
    auto e_tmp_indices = EigenMatrix<int64_t>::From(temp_indices);

    std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(k)};
    auto dim = framework::make_ddim(odims);
    auto e_values = EigenMatrix<T>::From(*out_tensor, dim);
    auto e_tmp_values = EigenMatrix<T>::From(temp_values);

    e_indices.device(dev) = e_tmp_indices.slice(slice_indices, slice_sizes);
    e_values.device(dev) = e_tmp_values.slice(slice_indices, slice_sizes);
  }
  return true;
}

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

template <typename DeviceContext, typename T>
class TopkOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    int k = static_cast<int>(ctx.Attr<int>("k"));

    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      Tensor k_host;
      framework::TensorCopySync(*k_t, platform::CPUPlace(), &k_host);
      k = k_host.data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[output_dims.size() - 1] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    // FIXME(typhoonzero): data is always converted to type T?

    framework::DDim inputdims = input->dims();
    const int64_t input_height = framework::product(
        framework::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const int64_t input_width = inputdims[inputdims.size() - 1];
    const auto& dev_ctx = ctx.cuda_device_context();

    if ((input_width <= 1024 || k >= 128 || k == input_width)) {
      if (SortTopk<T>(dev_ctx, input, input_width, input_height, k, output,
                      indices)) {
        // Successed, return.
        return;
      } else {
        LOG(INFO) << "TopKOP: Some errors happened when use cub sorting, use "
                     "default topk kernel.";
      }
    }
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());
    if (k > input_width) k = input_width;

    // NOTE: pass lds and dim same to input width.
    // NOTE: old matrix implementation of stride is different to eigen.
    // TODO(typhoonzero): refine this kernel.
    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    switch (GetDesiredBlockDim(input_width)) {
      FIXED_BLOCK_DIM(
          KeMatrixTopK<T, 5,
                       kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
              output_data, k, indices_data, input_data, input_width,
              input_width, static_cast<int>(k), gridx, input_height));
      default:
        PADDLE_THROW("Error");
    }
  }
};

template <typename DeviceContext, typename T>
class TopkOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    auto* x = context.Input<Tensor>("X");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();
    size_t k = indices->dims()[indices->dims().size() - 1];

    framework::DDim xdims = x->dims();
    const size_t row =
        framework::product(framework::slice_ddim(xdims, 0, xdims.size() - 1));
    const size_t col = xdims[xdims.size() - 1];
    const auto& dev_ctx = context.cuda_device_context();

    const int kMaxHeight = 2048;
    int gridx = row < kMaxHeight ? row : kMaxHeight;
    switch (GetDesiredBlockDim(col)) {
      FIXED_BLOCK_DIM(
          AssignGrad<T, 5,
                     kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
              x_grad_data, indices_data, out_grad_data, row, col, k));
      default:
        PADDLE_THROW(
            platform::errors::Unavailable("Error occurs when Assign Grad."));
    }
  }
};
#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    top_k,
    paddle::operators::TopkOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    paddle::operators::TopkOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    paddle::operators::TopkOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    paddle::operators::TopkOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>,
    paddle::operators::TopkOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(
    top_k_grad,
    paddle::operators::TopkOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                            float>,
    paddle::operators::TopkOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                            double>,
    paddle::operators::TopkOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                            int>,
    paddle::operators::TopkOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                            int64_t>,
    paddle::operators::TopkOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                            paddle::platform::float16>);
