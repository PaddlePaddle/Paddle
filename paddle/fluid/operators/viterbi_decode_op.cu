/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/viterbi_decode_op.h"

#include "paddle/fluid/operators/arg_min_max_op_base.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/transpose_op.cu.h"

namespace paddle {
namespace operators {

#define CUDA_ARGMAX(kBlockDim)                                           \
  do {                                                                   \
    ArgmaxCUDAKernel<T, IndType,                                         \
                     kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>( \
        height, width, post, in_data, out_idx_data, out_data);           \
  } while (0)

#define DEFINE_CUDA_FUNTOR(functor_type, expr)            \
  template <typename T>                                   \
  struct Cuda##functor_type##Functor {                    \
    inline HOSTDEVICE T operator()(const T* args) const { \
      return args[0] expr args[1];                        \
    }                                                     \
  }

DEFINE_CUDA_FUNTOR(Add, +);
DEFINE_CUDA_FUNTOR(Sub, -);
DEFINE_CUDA_FUNTOR(Mul, *);

#define CUDA_ELEMENT_BINARY_OP(lhs, rhs, output, functor_type, dtype)    \
  do {                                                                   \
    std::vector<const Tensor*> ins{&lhs, &rhs};                          \
    std::vector<Tensor*> outs{&output};                                  \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, dtype, dtype>( \
        dev_ctx, ins, &outs, -1, Cuda##functor_type##Functor<dtype>());  \
  } while (0)

#define CUDA_ADD(lhs, rhs, output, dtype) \
  CUDA_ELEMENT_BINARY_OP(lhs, rhs, output, Add, dtype)

#define CUDA_SUB(lhs, rhs, output, dtype) \
  CUDA_ELEMENT_BINARY_OP(lhs, rhs, output, Sub, dtype)

#define CUDA_MUL(lhs, rhs, output, dtype) \
  CUDA_ELEMENT_BINARY_OP(lhs, rhs, output, Mul, dtype)

#define DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(func, op) \
  template <typename T, typename Enable = void>               \
  struct func {                                               \
    using ELEMENT_TYPE = T;                                   \
    inline HOSTDEVICE bool operator()(const T* args) const {  \
      return args[0] op args[1];                              \
    }                                                         \
  }

#define FIX_BLOCKDIM_CASE(block_dim) \
  case (block_dim):                  \
    CUDA_ARGMAX((block_dim));        \
    break

#define GET_MASK(lhs, rhs, mask, functor_template, dtype)                  \
  do {                                                                     \
    std::vector<const Tensor*> ins = {&lhs, &rhs};                         \
    std::vector<Tensor*> outs = {&mask};                                   \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, int64_t, dtype>( \
        dev_ctx, ins, &outs, -1, functor_template<int64_t>());             \
  } while (0)

DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaEqualFunctor, ==);
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaGreaterThanFunctor, >);
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaGreaterEqualFunctor, >=);

template <typename T, typename IndType, size_t BlockDim>
__global__ void ArgmaxCUDAKernel(const int64_t height,     // n * h
                                 const int64_t width,      // c
                                 const int64_t post_size,  // h
                                 const T* in, IndType* out_idx, T* out) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  cub::ArgMax reducer;
  T init = std::numeric_limits<T>::lowest();
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
      out_idx[idx] = static_cast<IndType>(kv_pair.key);
      out[idx] = kv_pair.value;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ARange(T* data, int end, T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int start = idx; idx < end; idx += gridDim.x) {
    data[idx] = idx * scale;
  }
}

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
  return 8;
}

template <typename T, typename IndType>
struct CUDAArgmax {
  void operator()(const framework::ExecutionContext& ctx, const Tensor& input,
                  Tensor* out_idx, Tensor* out, int axis) {
    // axis should be larger than or equals to 0
    framework::DDim input_dims = input.dims();
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

    const auto& dev_ctx = ctx.cuda_device_context();
    auto cu_stream = dev_ctx.stream();
    int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize().x;
    int64_t height = pre * post;
    int64_t width = n;
    int64_t grid_size = height < max_grid_dimx ? height : max_grid_dimx;

    const T* in_data = input.data<T>();
    IndType* out_idx_data = out_idx->data<IndType>();
    T* out_data = out->data<T>();
    switch (ComputeBlockSize(width)) {
      FIX_BLOCKDIM_CASE(8);
      FIX_BLOCKDIM_CASE(16);
      FIX_BLOCKDIM_CASE(32);
      FIX_BLOCKDIM_CASE(64);
      FIX_BLOCKDIM_CASE(128);
      FIX_BLOCKDIM_CASE(256);
      FIX_BLOCKDIM_CASE(512);
      FIX_BLOCKDIM_CASE(1024);
    }
  }
};

template <typename DeviceContext, typename T>
class ViterbiDecodeGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* length = ctx.Input<Tensor>("Length");
    auto* transition = ctx.Input<Tensor>("Transition");
    bool with_start_stop_tag = ctx.Attr<bool>("with_start_stop_tag");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto curr_place = ctx.GetPlace();
    auto batch_size = static_cast<int>(input->dims()[0]);
    auto seq_len = static_cast<int>(input->dims()[1]);
    auto n_labels = static_cast<int>(input->dims()[2]);
    math::SetConstant<DeviceContext, T> float_functor;
    math::SetConstant<DeviceContext, int64_t> int_functor;
    std::vector<Tensor> historys;
    CUDAArgmax<T, int64_t> cuda_argmax;
    // Create a large int data buffer
    int buffer_size = batch_size * seq_len + batch_size * n_labels * seq_len +
                      7 * batch_size + 10;
    CREATE_TENSOR(int_buffer, int64_t, buffer_size);
    TensorBuffer int_tensor_buffer(int_buffer);
    // Create a large float data buffer
    buffer_size = seq_len * batch_size * n_labels + 5 * batch_size * n_labels +
                  2 * n_labels * n_labels + batch_size * n_labels * n_labels +
                  2 * batch_size + 1;
    CREATE_TENSOR(float_buffer, T, buffer_size);
    TensorBuffer float_tensor_buffer(float_buffer);

    Tensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);

    auto* scores = ctx.Output<Tensor>("Scores");
    scores->mutable_data<T>(curr_place);

    auto out_idx_data = int_tensor_buffer.GetBufferBlock({1});
    auto out_data = int_tensor_buffer.GetBufferBlock({1});
    ArgmaxCUDAKernel<int64_t, int64_t, 32><<<1, 32, 0, dev_ctx.stream()>>>(
        1, left_length.numel(), 1, left_length.data<int64_t>(),
        out_idx_data.data<int64_t>(), out_data.data<int64_t>());
    Tensor max_seq_len_tenor;
    framework::TensorCopy(out_data, platform::CPUPlace(), &max_seq_len_tenor);
    int64_t max_seq_len = max_seq_len_tenor.data<int64_t>()[0];
    auto* path = ctx.Output<Tensor>("Path");
    path->Resize({batch_size, max_seq_len});
    path->mutable_data<int64_t>(curr_place);

    Tensor tpath = int_tensor_buffer.GetBufferBlock({max_seq_len, batch_size});
    auto batch_path = Unbind(tpath);
    for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
      it->Resize({batch_size});
    }
    Tensor inputs_t_exp =
        float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    TransposeGPUKernelDriver<T>(dev_ctx, 3, *input, {1, 0, 2}, &inputs_t_exp);
    Tensor trans_exp =
        float_tensor_buffer.GetBufferBlock({1, n_labels, n_labels});
    framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);
    Tensor alpha = float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor zero = int_tensor_buffer.GetBufferBlock({1});
    int_functor(dev_ctx, &zero, 0);
    Tensor one = int_tensor_buffer.GetBufferBlock({1});
    int_functor(dev_ctx, &one, 1);
    Tensor float_one = float_tensor_buffer.GetBufferBlock({batch_size, 1});
    float_functor(dev_ctx, &float_one, static_cast<T>(1.0));
    Tensor alpha_trn_sum =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels, n_labels});
    Tensor alpha_max =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor alpha_argmax =
        int_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    auto alpha_argmax_unbind = Unbind(alpha_argmax);
    Tensor alpha_nxt =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor int_mask = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor float_mask = float_tensor_buffer.GetBufferBlock({batch_size, 1});
    Tensor stop_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor start_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor rest_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, n_labels - 2, n_labels});
    Tensor last_ids = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor batch_offset = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor gather_idx = int_tensor_buffer.GetBufferBlock({batch_size});

    std::vector<const Tensor*> shape_refer{&rest_trans_exp, &stop_trans_exp,
                                           &start_trans_exp};
    std::vector<Tensor*> outputs{&rest_trans_exp, &stop_trans_exp,
                                 &start_trans_exp};
    math::SplitFunctor<DeviceContext, T> split_functor;
    split_functor(dev_ctx, trans_exp, shape_refer, 1, &outputs);
    stop_trans_exp.Resize({1, n_labels});
    start_trans_exp.Resize({1, n_labels});
    auto logit0 = inputs_t_exp.Slice(0, 1);
    logit0.Resize({batch_size, n_labels});
    if (with_start_stop_tag) {
      CUDA_ADD(logit0, start_trans_exp, alpha, T);
      GET_MASK(left_length, one, float_mask, CudaEqualFunctor, T);
      CUDA_MUL(stop_trans_exp, float_mask, alpha_nxt, T);
      CUDA_ADD(alpha, alpha_nxt, alpha, T);
    } else {
      alpha = logit0;
    }
    CUDA_SUB(left_length, one, left_length, int64_t);
    for (int64_t i = 1; i < max_seq_len; ++i) {
      Tensor logit = inputs_t_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});
      CUDA_ADD(alpha_exp, trans_exp, alpha_trn_sum, T);
      auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
      alpha_argmax_temp.Resize({batch_size, n_labels});
      cuda_argmax(ctx, alpha_trn_sum, &alpha_argmax_temp, &alpha_max, 1);
      historys.push_back(alpha_argmax_temp);
      CUDA_ADD(alpha_max, logit, alpha_nxt, T);
      alpha.Resize({batch_size, n_labels});
      GET_MASK(left_length, zero, float_mask, CudaGreaterThanFunctor, T);
      CUDA_MUL(alpha_nxt, float_mask, alpha_nxt, T);
      CUDA_SUB(float_one, float_mask, float_mask, T);
      CUDA_MUL(alpha, float_mask, alpha, T);
      CUDA_ADD(alpha, alpha_nxt, alpha, T);
      if (with_start_stop_tag) {
        GET_MASK(left_length, one, float_mask, CudaEqualFunctor, T);
        CUDA_MUL(stop_trans_exp, float_mask, alpha_nxt, T);
        CUDA_ADD(alpha, alpha_nxt, alpha, T);
      }
      CUDA_SUB(left_length, one, left_length, int64_t);
    }
    cuda_argmax(ctx, alpha, &last_ids, scores, 1);
    left_length.Resize({batch_size});
    GET_MASK(left_length, zero, int_mask, CudaGreaterEqualFunctor, int64_t);
    int last_ids_index = 1;
    int actual_len = std::min(seq_len, static_cast<int>(max_seq_len));
    CUDA_MUL(last_ids, int_mask, batch_path[actual_len - last_ids_index],
             int64_t);
    auto block_size = ComputeBlockSize(batch_size);
    ARange<int64_t><<<1, block_size, 0, dev_ctx.stream()>>>(
        batch_offset.data<int64_t>(), batch_size, n_labels);

    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      ++last_ids_index;
      CUDA_ADD(left_length, one, left_length, int64_t);
      CUDA_ADD(batch_offset, last_ids, gather_idx, int64_t);
      Tensor& last_ids_update = batch_path[actual_len - last_ids_index];
      hist->Resize({batch_size * n_labels});
      GPUGather<int64_t, int64_t>(dev_ctx, *hist, gather_idx, &last_ids_update);
      GET_MASK(left_length, zero, int_mask, CudaGreaterEqualFunctor, int64_t);
      CUDA_MUL(last_ids_update, int_mask, last_ids_update, int64_t);
      CUDA_SUB(one, int_mask, int_mask, int64_t);
      CUDA_MUL(last_ids, int_mask, last_ids, int64_t);
      CUDA_ADD(last_ids_update, last_ids, last_ids, int64_t);
    }
    TransposeGPUKernelDriver<int64_t>(dev_ctx, 2, tpath, {1, 0}, path);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    viterbi_decode,
    ops::ViterbiDecodeGPUKernel<platform::CUDADeviceContext, float>,
    ops::ViterbiDecodeGPUKernel<platform::CUDADeviceContext, double>);
