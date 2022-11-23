// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/viterbi_decode_kernel.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/viterbi_decode_functor.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)               \
  FIXED_BLOCK_DIM_CASE_BASE(10, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__);

int64_t ComputeBlockSize(int64_t col) {
  if (col > 512)
    return 1024;
  else if (col > 256)
    return 512;
  else if (col > 128)
    return 256;
  else if (col > 64)
    return 128;
  else if (col > 32)
    return 64;
  else if (col > 16)
    return 32;
  else if (col > 8)
    return 16;
  else
    return 8;
}

template <typename Context,
          template <typename T>
          typename BinaryFunctor,
          typename T>
struct BinaryOperation {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& lhs,
                  const DenseTensor& rhs,
                  DenseTensor* output) {
    std::vector<const DenseTensor*> ins{&lhs, &rhs};
    std::vector<DenseTensor*> outs{output};
    paddle::operators::
        LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
            dev_ctx, ins, &outs, 0, BinaryFunctor<T>());
  }
};

template <typename Context,
          template <typename InT, typename OutT>
          typename CompareFunctor,
          typename T>
struct GetMask {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& lhs,
                  const DenseTensor& rhs,
                  DenseTensor* mask) {
    std::vector<const DenseTensor*> ins = {&lhs, &rhs};
    std::vector<DenseTensor*> outs = {mask};
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
        dev_ctx, ins, &outs, CompareFunctor<int64_t, T>());
  }
};

template <typename T, typename IndType, size_t BlockDim>
__global__ void ArgmaxCUDAKernel(const int64_t height,     // n * h
                                 const int64_t width,      // c
                                 const int64_t post_size,  // h
                                 const T* in,
                                 IndType* out_idx,
                                 T* out) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  cub::ArgMax reducer;
  T init = (std::numeric_limits<T>::lowest)();  // for windows compile
  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    cub::KeyValuePair<int, T> kv_pair = {-1, init};
    int h = idx / post_size;
    int w = idx % post_size;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair =
          reducer({k, in[h * width * post_size + k * post_size + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      // return max, argmax
      if (out_idx != nullptr) out_idx[idx] = static_cast<IndType>(kv_pair.key);
      if (out != nullptr) out[idx] = kv_pair.value;
    }
    __syncthreads();
  }
}

__global__ void ARangeKernel(int64_t* data, int num, int64_t scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int start = idx; idx < num; idx += gridDim.x) {
    data[idx] = idx * scale;
  }
}

template <typename Context>
struct ARange {
  void operator()(const Context& dev_ctx,
                  int64_t* data,
                  int num,
                  int64_t scale) {
    int64_t kBlockDim = ComputeBlockSize(num);
    // kBlockDim > num at most of time, so we can set grid = 1
    ARangeKernel<<<1, kBlockDim, 0, dev_ctx.stream()>>>(data, num, scale);
  }
};

template <typename Context, typename T, typename IndType>
struct Argmax {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* out_idx,
                  DenseTensor* out,
                  int axis) {
    phi::DDim input_dims = input.dims();
    int64_t numel = input.numel();
    int64_t groups = numel / input_dims[axis];
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = input_dims[axis];
    for (int i = 0; i < axis; i++) {
      pre *= input_dims[i];
    }
    for (int i = axis + 1; i < input_dims.size(); i++) {
      post *= input_dims[i];
    }
    auto cu_stream = dev_ctx.stream();
    int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int64_t height = pre * post;
    int64_t width = n;
    int64_t grid_size = height < max_grid_dimx ? height : max_grid_dimx;
    const T* in_data = input.data<T>();
    IndType* out_idx_data = out_idx->data<IndType>();
    T* out_data = out->data<T>();
    switch (ComputeBlockSize(width)) {
      FIXED_BLOCK_DIM_CASE(
          ArgmaxCUDAKernel<T, IndType, kBlockDim>
          <<<grid_size, kBlockDim, 0, cu_stream>>>(
              height, width, post, in_data, out_idx_data, out_data));
    }
  }
};

template <typename Context, typename T>
struct GetMaxValue {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  T* max_value) {
    DenseTensor out_data;
    out_data.Resize(phi::make_ddim({1}));
    dev_ctx.template Alloc<T>(&out_data);
    switch (ComputeBlockSize(input.numel())) {
      FIXED_BLOCK_DIM_CASE(
          ArgmaxCUDAKernel<T, T, kBlockDim>
          <<<1, kBlockDim, 0, dev_ctx.stream()>>>(1,
                                                  input.numel(),
                                                  1,
                                                  input.data<int64_t>(),
                                                  nullptr,
                                                  out_data.data<int64_t>()));
    }
    DenseTensor max_value_tensor;
    phi::Copy(dev_ctx, out_data, phi::CPUPlace(), false, &max_value_tensor);
    *max_value = max_value_tensor.data<T>()[0];
  }
};

template <typename Context, typename T, typename IndexT>
struct Gather {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& src,
                  const DenseTensor& index,
                  DenseTensor* output) {
    phi::funcs::GPUGather<T, IndexT>(dev_ctx, src, index, output);
  }
};

template <typename T, typename Context>
void ViterbiDecodeKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& transition,
                         const DenseTensor& length,
                         bool include_bos_eos_tag,
                         DenseTensor* scores,
                         DenseTensor* path) {
  auto curr_place = dev_ctx.GetPlace();
  auto batch_size = static_cast<int>(input.dims()[0]);
  auto seq_len = static_cast<int>(input.dims()[1]);
  auto n_labels = static_cast<int>(input.dims()[2]);
  phi::funcs::SetConstant<Context, T> float_functor;
  phi::funcs::SetConstant<Context, int64_t> int_functor;
  std::vector<DenseTensor> historys;
  // We create tensor buffer in order to avoid allocating memory frequently
  // 10 means allocate 10*batch_size bytes memory, such as int_mask, zero...
  int buffer_size = batch_size * (n_labels + 1) * seq_len + 10 * batch_size;
  DenseTensor int_buffer = Empty<int64_t>(dev_ctx, {buffer_size});
  funcs::TensorBuffer int_tensor_buffer(int_buffer);
  // create float tensor buffer
  // 10 means allocate 10*batch_size*n_labels bytes, such as alpha, alpha_max
  buffer_size = batch_size * (seq_len + 10) * n_labels +
                (batch_size + 2) * n_labels * n_labels;
  DenseTensor float_buffer = Empty<T>(dev_ctx, {buffer_size});
  funcs::TensorBuffer float_tensor_buffer(float_buffer);
  DenseTensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
  phi::Copy(dev_ctx, length, curr_place, false, &left_length);
  int64_t max_seq_len = 0;
  GetMaxValue<Context, int64_t> get_max_value;
  get_max_value(dev_ctx, left_length, &max_seq_len);
  dev_ctx.template Alloc<T>(scores);
  path->Resize({batch_size, max_seq_len});
  dev_ctx.template Alloc<int64_t>(path);
  DenseTensor tpath =
      int_tensor_buffer.GetBufferBlock({max_seq_len, batch_size});
  auto batch_path = funcs::Unbind(tpath);
  for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
    it->Resize({batch_size});
  }
  // create and init required tensor
  DenseTensor input_exp =
      float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
  TransposeKernel<T, Context>(dev_ctx, input, {1, 0, 2}, &input_exp);
  DenseTensor trans_exp =
      float_tensor_buffer.GetBufferBlock({n_labels, n_labels});
  phi::Copy(dev_ctx, transition, curr_place, false, &trans_exp);
  trans_exp.Resize({1, n_labels, n_labels});
  DenseTensor alpha =
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
  DenseTensor zero = int_tensor_buffer.GetBufferBlock({batch_size, 1});
  int_functor(dev_ctx, &zero, 0);
  DenseTensor one = int_tensor_buffer.GetBufferBlock({batch_size, 1});
  int_functor(dev_ctx, &one, 1);
  DenseTensor float_one = float_tensor_buffer.GetBufferBlock({batch_size, 1});
  float_functor(dev_ctx, &float_one, static_cast<T>(1.0));
  DenseTensor alpha_trn_sum =
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels, n_labels});
  DenseTensor alpha_max =
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
  DenseTensor alpha_argmax =
      int_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
  auto alpha_argmax_unbind = funcs::Unbind(alpha_argmax);
  DenseTensor alpha_nxt =
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
  DenseTensor int_mask = int_tensor_buffer.GetBufferBlock({batch_size});
  DenseTensor zero_len_mask = int_tensor_buffer.GetBufferBlock({batch_size});
  DenseTensor float_mask = float_tensor_buffer.GetBufferBlock({batch_size, 1});
  DenseTensor stop_trans = float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
  DenseTensor start_trans =
      float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
  DenseTensor rest_trans =
      float_tensor_buffer.GetBufferBlock({1, n_labels - 2, n_labels});
  DenseTensor last_ids = int_tensor_buffer.GetBufferBlock({batch_size});
  DenseTensor last_ids_tmp = int_tensor_buffer.GetBufferBlock({batch_size});
  DenseTensor batch_offset = int_tensor_buffer.GetBufferBlock({batch_size});
  DenseTensor gather_idx = int_tensor_buffer.GetBufferBlock({batch_size});
  std::vector<const DenseTensor*> shape{&rest_trans, &stop_trans, &start_trans};
  std::vector<DenseTensor*> outputs{&rest_trans, &stop_trans, &start_trans};
  phi::funcs::SplitFunctor<Context, T> split_functor;
  split_functor(dev_ctx, trans_exp, shape, 1, &outputs);
  stop_trans.Resize({1, n_labels});
  start_trans.Resize({1, n_labels});
  auto logit0 = input_exp.Slice(0, 1);
  logit0.Resize({batch_size, n_labels});
  BinaryOperation<Context, phi::funcs::AddFunctor, T> AddFloat;
  BinaryOperation<Context, phi::funcs::AddFunctor, int64_t> AddInt;
  BinaryOperation<Context, phi::funcs::MultiplyFunctor, T> MulFloat;
  BinaryOperation<Context, phi::funcs::MultiplyFunctor, int64_t> MulInt;
  BinaryOperation<Context, phi::funcs::SubtractFunctor, T> SubFloat;
  BinaryOperation<Context, phi::funcs::SubtractFunctor, int64_t> SubInt;
  if (include_bos_eos_tag) {
    AddFloat(dev_ctx, logit0, start_trans, &alpha);
    GetMask<Context, phi::funcs::EqualFunctor, T>()(
        dev_ctx, left_length, one, &float_mask);
    MulFloat(dev_ctx, stop_trans, float_mask, &alpha_nxt);
    AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
  } else {
    alpha = logit0;
  }
  SubInt(dev_ctx, left_length, one, &left_length);
  Argmax<Context, T, int64_t> argmax;
  for (int64_t i = 1; i < max_seq_len; ++i) {
    DenseTensor logit = input_exp.Slice(i, i + 1);
    logit.Resize({batch_size, n_labels});
    DenseTensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});
    AddFloat(dev_ctx, alpha_exp, trans_exp, &alpha_trn_sum);
    auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
    alpha_argmax_temp.Resize({batch_size, n_labels});
    argmax(dev_ctx, alpha_trn_sum, &alpha_argmax_temp, &alpha_max, 1);
    historys.emplace_back(alpha_argmax_temp);
    AddFloat(dev_ctx, alpha_max, logit, &alpha_nxt);
    alpha.Resize({batch_size, n_labels});
    GetMask<Context, phi::funcs::GreaterThanFunctor, T>()(
        dev_ctx, left_length, zero, &float_mask);
    MulFloat(dev_ctx, alpha_nxt, float_mask, &alpha_nxt);
    SubFloat(dev_ctx, float_one, float_mask, &float_mask);
    MulFloat(dev_ctx, alpha, float_mask, &alpha);
    AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
    if (include_bos_eos_tag) {
      GetMask<Context, phi::funcs::EqualFunctor, T>()(
          dev_ctx, left_length, one, &float_mask);
      MulFloat(dev_ctx, stop_trans, float_mask, &alpha_nxt);
      AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
    }
    SubInt(dev_ctx, left_length, one, &left_length);
  }
  argmax(dev_ctx, alpha, &last_ids, scores, 1);
  left_length.Resize({batch_size});
  GetMask<Context, phi::funcs::GreaterEqualFunctor, int64_t>()(
      dev_ctx, left_length, zero, &int_mask);
  // last_ids_update = last_ids * tag_mask
  int last_ids_index = 1;
  int actual_len = (std::min)(seq_len, static_cast<int>(max_seq_len));
  MulInt(dev_ctx, last_ids, int_mask, &batch_path[actual_len - last_ids_index]);
  // The algorithm below can refer to
  // https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/layers/crf.py#L438
  ARange<Context> arange;
  arange(dev_ctx, batch_offset.data<int64_t>(), batch_size, n_labels);
  Gather<Context, int64_t, int64_t> gather;
  for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
    ++last_ids_index;
    AddInt(dev_ctx, left_length, one, &left_length);
    AddInt(dev_ctx, batch_offset, last_ids, &gather_idx);
    DenseTensor& last_ids_update = batch_path[actual_len - last_ids_index];
    hist->Resize({batch_size * n_labels});
    gather(dev_ctx, *hist, gather_idx, &last_ids_update);
    GetMask<Context, phi::funcs::GreaterThanFunctor, int64_t>()(
        dev_ctx, left_length, zero, &int_mask);
    MulInt(dev_ctx, last_ids_update, int_mask, &last_ids_update);
    GetMask<Context, phi::funcs::EqualFunctor, int64_t>()(
        dev_ctx, left_length, zero, &zero_len_mask);
    MulInt(dev_ctx, last_ids, zero_len_mask, &last_ids_tmp);
    SubInt(dev_ctx, one, zero_len_mask, &zero_len_mask);
    MulInt(dev_ctx, last_ids_update, zero_len_mask, &last_ids_update);
    AddInt(dev_ctx, last_ids_update, last_ids_tmp, &last_ids_update);
    GetMask<Context, phi::funcs::LessThanFunctor, int64_t>()(
        dev_ctx, left_length, zero, &int_mask);
    MulInt(dev_ctx, last_ids, int_mask, &last_ids);
    AddInt(dev_ctx, last_ids_update, last_ids, &last_ids);
  }
  TransposeKernel<int64_t, Context>(dev_ctx, tpath, {1, 0}, path);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    viterbi_decode, GPU, ALL_LAYOUT, phi::ViterbiDecodeKernel, float, double) {}
