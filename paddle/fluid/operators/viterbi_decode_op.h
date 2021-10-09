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

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_functor.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unique_op.h"
#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

#define CREATE_TENSOR(tensor, dtype, ...)             \
  LoDTensor tensor;                                   \
  tensor.Resize(framework::make_ddim({__VA_ARGS__})); \
  tensor.mutable_data<dtype>(ctx.GetPlace())

#define CREATE_TENSOR_BUFFER(int_tensor_buffer, float_tensor_buffer)          \
  int buffer_size = batch_size * seq_len + batch_size * n_labels * seq_len +  \
                    9 * batch_size + 10;                                      \
  CREATE_TENSOR(int_buffer, int64_t, buffer_size);                            \
  TensorBuffer int_tensor_buffer(int_buffer);                                 \
  buffer_size = seq_len * batch_size * n_labels + 5 * batch_size * n_labels + \
                2 * n_labels * n_labels + batch_size * n_labels * n_labels +  \
                3 * batch_size + 1;                                           \
  CREATE_TENSOR(float_buffer, T, buffer_size);                                \
  TensorBuffer float_tensor_buffer(float_buffer)

#define INIT_REQUIRED_TENSOR()                                                 \
  Tensor input_exp =                                                           \
      float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});     \
  TransCompute<DeviceContext, T>(3, dev_ctx, *input, &input_exp, {1, 0, 2});   \
  auto* transition = ctx.Input<Tensor>("Transition");                          \
  Tensor trans_exp = float_tensor_buffer.GetBufferBlock({n_labels, n_labels}); \
  framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);         \
  trans_exp.Resize({1, n_labels, n_labels});                                   \
  Tensor alpha = float_tensor_buffer.GetBufferBlock({batch_size, n_labels});   \
  Tensor zero = int_tensor_buffer.GetBufferBlock({1});                         \
  int_functor(dev_ctx, &zero, 0);                                              \
  Tensor one = int_tensor_buffer.GetBufferBlock({1});                          \
  int_functor(dev_ctx, &one, 1);                                               \
  Tensor float_one = float_tensor_buffer.GetBufferBlock({batch_size, 1});      \
  float_functor(dev_ctx, &float_one, static_cast<T>(1.0));                     \
  Tensor alpha_trn_sum =                                                       \
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels, n_labels});    \
  Tensor alpha_max =                                                           \
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels});              \
  Tensor alpha_argmax =                                                        \
      int_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});       \
  auto alpha_argmax_unbind = Unbind(alpha_argmax);                             \
  Tensor alpha_nxt =                                                           \
      float_tensor_buffer.GetBufferBlock({batch_size, n_labels});              \
  Tensor int_mask = int_tensor_buffer.GetBufferBlock({batch_size});            \
  Tensor zero_len_mask = int_tensor_buffer.GetBufferBlock({batch_size});       \
  Tensor float_mask = float_tensor_buffer.GetBufferBlock({batch_size, 1});     \
  Tensor stop_trans_exp =                                                      \
      float_tensor_buffer.GetBufferBlock({1, 1, n_labels});                    \
  Tensor start_trans_exp =                                                     \
      float_tensor_buffer.GetBufferBlock({1, 1, n_labels});                    \
  Tensor rest_trans_exp =                                                      \
      float_tensor_buffer.GetBufferBlock({1, n_labels - 2, n_labels});         \
  Tensor last_ids = int_tensor_buffer.GetBufferBlock({batch_size});            \
  Tensor last_ids_tmp = int_tensor_buffer.GetBufferBlock({batch_size});        \
  Tensor batch_offset = int_tensor_buffer.GetBufferBlock({batch_size});        \
  Tensor gather_idx = int_tensor_buffer.GetBufferBlock({batch_size});          \
  std::vector<const Tensor*> shape_refer{&rest_trans_exp, &stop_trans_exp,     \
                                         &start_trans_exp};                    \
  std::vector<Tensor*> outputs{&rest_trans_exp, &stop_trans_exp,               \
                               &start_trans_exp};                              \
  math::SplitFunctor<DeviceContext, T> split_functor;                          \
  split_functor(dev_ctx, trans_exp, shape_refer, 1, &outputs);                 \
  stop_trans_exp.Resize({1, n_labels});                                        \
  start_trans_exp.Resize({1, n_labels});                                       \
  auto logit0 = input_exp.Slice(0, 1);                                         \
  logit0.Resize({batch_size, n_labels})

#define BROADCAST_BINARY_OP(lhs, rhs, out, func_type, is_multi_threads, dtype) \
  do {                                                                         \
    if (is_multi_threads) {                                                    \
      SimpleBroadcastBinaryOP<dtype, func_type##Functor<dtype>, true>(         \
          lhs, rhs, &out);                                                     \
    } else {                                                                   \
      SimpleBroadcastBinaryOP<dtype, func_type##Functor<dtype>, false>(        \
          lhs, rhs, &out);                                                     \
    }                                                                          \
  } while (0)

#define ADD(lhs, rhs, output, is_multi_threads, dtype) \
  BROADCAST_BINARY_OP(lhs, rhs, output, Add, is_multi_threads, dtype)

#define SUB(lhs, rhs, output, is_multi_threads, dtype) \
  BROADCAST_BINARY_OP(lhs, rhs, output, Sub, is_multi_threads, dtype)

#define MUL(lhs, rhs, output, is_multi_threads, dtype) \
  BROADCAST_BINARY_OP(lhs, rhs, output, Mul, is_multi_threads, dtype)

#define GET_MASK(lhs, rhs, mask, functor_template, dtype)                 \
  ElementwiseComputeEx<functor_template<int64_t>, DeviceContext, int64_t, \
                       dtype>(ctx, &lhs, &rhs, -1,                        \
                              functor_template<int64_t>(), &mask)

template <typename T, typename IndType>
struct CPUArgmax {
  void operator()(const Tensor& input, Tensor* out_idx, Tensor* out, int axis) {
    framework::DDim input_dims = input.dims();
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = input_dims[axis];
    for (int i = 0; i < axis; i++) {
      pre *= input_dims[i];
    }
    for (int i = axis + 1; i < input_dims.size(); i++) {
      post *= input_dims[i];
    }
    int64_t height = pre * post;
    int64_t width = n;
    const T* in_data = input.data<T>();
    IndType* out_idx_data = out_idx->data<IndType>();
    T* out_data = out->data<T>();
// Reduce
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < height; ++i) {
      int64_t h = i / post;
      int64_t w = i % post;
      IndType max_idx = -1;
      T max_value = (std::numeric_limits<T>::lowest)();  // for windows compile
      for (int64_t j = 0; j < width; ++j) {
        if (in_data[h * width * post + j * post + w] > max_value) {
          max_value = in_data[h * width * post + j * post + w];
          max_idx = j;
        }
      }
      out_data[i] = max_value;
      out_idx_data[i] = max_idx;
    }
  }
};

template <typename T, typename Functor>
void SameDimsBinaryOP(const Tensor& lhs, const Tensor& rhs, Tensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  T* out_ptr = out->data<T>();
  Functor functor;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < out->numel(); ++i) {
    out_ptr[i] = functor(lhs_ptr[i], rhs_ptr[i]);
  }
}

// Need to gurantee that lhs, rhs have same dims.
#define SAME_DIMS_OP(lhs, rhs, output, functor_type, dtype) \
  SameDimsBinaryOP<dtype, functor_type##Functor<dtype>>(lhs, rhs, &output)

template <bool is_multi_threads>
struct GetInputIndex {
  void operator()(const std::vector<int>& lhs_dims,
                  const std::vector<int>& rhs_dims,
                  const std::vector<int>& output_dims,
                  const std::vector<int>& lhs_strides,
                  const std::vector<int>& rhs_strides,
                  const std::vector<int>& output_strides, int output_idx,
                  int* index_array, int* lhs_idx, int* rhs_idx) {
    int out_dims_size = output_strides.size();
    for (int j = 0; j < out_dims_size; ++j) {
      int curr_idx = output_idx / output_strides[j];
      output_idx %= output_strides[j];
      *lhs_idx += (lhs_dims[j] > 1) ? curr_idx * lhs_strides[j] : 0;
      *rhs_idx += (rhs_dims[j] > 1) ? curr_idx * rhs_strides[j] : 0;
    }
  }
};

template <>
struct GetInputIndex<false> {
  void operator()(const std::vector<int>& lhs_dims,
                  const std::vector<int>& rhs_dims,
                  const std::vector<int>& output_dims,
                  const std::vector<int>& lhs_strides,
                  const std::vector<int>& rhs_strides,
                  const std::vector<int>& output_strides, int output_idx,
                  int* index_array, int* lhs_idx, int* rhs_idx) {
    int out_dims_size = output_strides.size();
    *lhs_idx = GetElementwiseIndex(lhs_dims.data(), out_dims_size, index_array);
    *rhs_idx = GetElementwiseIndex(rhs_dims.data(), out_dims_size, index_array);
    UpdateElementwiseIndexArray(output_dims.data(), out_dims_size, index_array);
  }
};

template <typename T, typename Functor, bool is_multi_threads = false>
void SimpleBroadcastBinaryOP(const Tensor& lhs, const Tensor& rhs,
                             Tensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  T* out_ptr = out->data<T>();
  int out_size = static_cast<int>(out->dims().size());
  std::vector<int> out_dims(out_size);
  std::vector<int> lhs_dims(out_size);
  std::vector<int> rhs_dims(out_size);
  std::copy(lhs.dims().Get(), lhs.dims().Get() + out_size, lhs_dims.data());
  std::copy(rhs.dims().Get(), rhs.dims().Get() + out_size, rhs_dims.data());
  std::copy(out->dims().Get(), out->dims().Get() + out_size, out_dims.data());
  std::vector<int> output_strides(out_size, 1);
  std::vector<int> lhs_strides(out_size, 1);
  std::vector<int> rhs_strides(out_size, 1);
  std::vector<int> index_array(out_size, 0);
  // calculate strides
  for (int i = out_size - 2; i >= 0; --i) {
    output_strides[i] = output_strides[i + 1] * out_dims[i + 1];
    lhs_strides[i] = lhs_strides[i + 1] * lhs_dims[i + 1];
    rhs_strides[i] = rhs_strides[i + 1] * rhs_dims[i + 1];
  }
  Functor functor;
  GetInputIndex<is_multi_threads> get_input_index;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < out->numel(); ++i) {
    int lhs_idx = 0;
    int rhs_idx = 0;
    get_input_index(lhs_dims, rhs_dims, out_dims, lhs_strides, rhs_strides,
                    output_strides, i, index_array.data(), &lhs_idx, &rhs_idx);
    out_ptr[i] = functor(lhs_ptr[lhs_idx], rhs_ptr[rhs_idx]);
  }
}

class TensorBuffer {
 public:
  explicit TensorBuffer(const LoDTensor& in) : buffer_(in), offset_(0) {
    buffer_.Resize({buffer_.numel()});
  }
  Tensor GetBufferBlock(std::initializer_list<int64_t> shape) {
    int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
    Tensor block = buffer_.Slice(offset_, offset_ + size);
    offset_ += size;
    block.Resize(shape);
    return block;
  }

 private:
  LoDTensor buffer_;  // need to resize 1-D Tensor
  int offset_;
};

template <typename DeviceContext, typename T>
class ViterbiDecodeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool with_start_stop_tag = ctx.Attr<bool>("with_start_stop_tag");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto curr_place = ctx.GetPlace();
    auto* input = ctx.Input<Tensor>("Input");
    auto batch_size = static_cast<int>(input->dims()[0]);
    auto seq_len = static_cast<int>(input->dims()[1]);
    auto n_labels = static_cast<int>(input->dims()[2]);
    math::SetConstant<DeviceContext, T> float_functor;
    math::SetConstant<DeviceContext, int64_t> int_functor;
    std::vector<Tensor> historys;
    CREATE_TENSOR_BUFFER(int_tensor_buffer, float_tensor_buffer);
    auto* length = ctx.Input<Tensor>("Length");
    Tensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);
    auto len_ptr = left_length.data<int64_t>();
    int64_t max_seq_len = *std::max_element(len_ptr, len_ptr + batch_size);

    auto* scores = ctx.Output<Tensor>("Scores");
    scores->mutable_data<T>(curr_place);
    auto* path = ctx.Output<Tensor>("Path");
    path->Resize({batch_size, max_seq_len});
    path->mutable_data<int64_t>(curr_place);
    Tensor tpath = int_tensor_buffer.GetBufferBlock({max_seq_len, batch_size});
    auto batch_path = Unbind(tpath);
    for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
      it->Resize({batch_size});
    }
    // create and init required tensor
    INIT_REQUIRED_TENSOR();
    bool is_multi_threads = false;
#ifdef PADDLE_WITH_MKLML
    if (omp_get_max_threads() > 1) {
      is_multi_threads = true;
    }
#endif
    if (with_start_stop_tag) {
      ADD(logit0, start_trans_exp, alpha, is_multi_threads, T);
      GET_MASK(left_length, one, float_mask, EqualFunctor, T);
      MUL(stop_trans_exp, float_mask, alpha_nxt, is_multi_threads, T);
      SAME_DIMS_OP(alpha, alpha_nxt, alpha, Add, T);
    } else {
      alpha = logit0;
    }
    SUB(left_length, one, left_length, is_multi_threads, int64_t);
    CPUArgmax<T, int64_t> argmax;
    for (int64_t i = 1; i < max_seq_len; ++i) {
      Tensor logit = input_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});
      ADD(alpha_exp, trans_exp, alpha_trn_sum, is_multi_threads, T);
      auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
      alpha_argmax_temp.Resize({batch_size, n_labels});
      argmax(alpha_trn_sum, &alpha_argmax_temp, &alpha_max, 1);
      historys.push_back(alpha_argmax_temp);
      SAME_DIMS_OP(alpha_max, logit, alpha_nxt, Add, T);
      alpha.Resize({batch_size, n_labels});
      // mask = paddle.cast((left_length > 0), dtype='float32')
      // alpha = mask * alpha_nxt + (1 - mask) * alpha
      GET_MASK(left_length, zero, float_mask, GreaterThanFunctor, T);
      // alpha_nxt = mask * alpha_nxt
      MUL(alpha_nxt, float_mask, alpha_nxt, is_multi_threads, T);
      // inv_mask = 1 - mask
      SAME_DIMS_OP(float_one, float_mask, float_mask, Sub, T);
      // alpha = (1 - mask) * alpha
      MUL(alpha, float_mask, alpha, is_multi_threads, T);
      // alpha += alpha_nxt
      SAME_DIMS_OP(alpha, alpha_nxt, alpha, Add, T);
      if (with_start_stop_tag) {  // cost 10% time
        GET_MASK(left_length, one, float_mask, EqualFunctor, T);
        // trans_exp: [1, n, n]
        // alpha += mask * trans_exp[:, self.stop_idx]
        MUL(stop_trans_exp, float_mask, alpha_nxt, is_multi_threads, T);
        SAME_DIMS_OP(alpha, alpha_nxt, alpha, Add, T);
      }
      SUB(left_length, one, left_length, is_multi_threads, int64_t);
    }
    // scores, last_ids = alpha.max(1), alpha.argmax(1)
    argmax(alpha, &last_ids, scores, 1);
    // tag_mask = paddle.cast((left_length >= 0), 'int64')
    left_length.Resize({batch_size});
    GET_MASK(left_length, zero, int_mask, GreaterEqualFunctor, int64_t);
    // last_ids_update = last_ids * tag_mask
    int last_ids_index = 1;
    int actual_len = (std::min)(seq_len, static_cast<int>(max_seq_len));

    SAME_DIMS_OP(last_ids, int_mask, batch_path[actual_len - last_ids_index],
                 Mul, int64_t);
    for (int i = 0; i < batch_size; ++i) {
      batch_offset.data<int64_t>()[i] = i * n_labels;
    }
    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      ++last_ids_index;
      ADD(left_length, one, left_length, is_multi_threads, int64_t);
      SAME_DIMS_OP(batch_offset, last_ids, gather_idx, Add, int64_t);
      // tag_mask = paddle.cast((left_length > 0), 'int64')
      // last_ids_update = paddle.gather(hist.flatten(), gather_idx) * tag_mask
      // zero_len_mask = paddle.cast((left_length == 0), 'int64')
      // last_ids_update = last_ids_update * (1 - zero_len_mask) + last_ids *
      // zero_len_mask
      Tensor& last_ids_update = batch_path[actual_len - last_ids_index];
      hist->Resize({batch_size * n_labels});
      CPUGather<int64_t, int64_t>(dev_ctx, *hist, gather_idx, &last_ids_update);
      GET_MASK(left_length, zero, int_mask, GreaterThanFunctor, int64_t);
      SAME_DIMS_OP(last_ids_update, int_mask, last_ids_update, Mul, int64_t);
      GET_MASK(left_length, zero, zero_len_mask, EqualFunctor, int64_t);
      SAME_DIMS_OP(last_ids, zero_len_mask, last_ids_tmp, Mul, int64_t);
      SUB(one, zero_len_mask, zero_len_mask, is_multi_threads, int64_t);
      SAME_DIMS_OP(last_ids_update, zero_len_mask, last_ids_update, Mul,
                   int64_t);
      SAME_DIMS_OP(last_ids_update, last_ids_tmp, last_ids_update, Add,
                   int64_t);
      GET_MASK(left_length, zero, int_mask, LessThanFunctor, int64_t);
      // tag_mask = paddle.cast((left_length < 0), 'int64');
      // last_ids = last_ids_update + last_ids * tag_mask
      SAME_DIMS_OP(last_ids, int_mask, last_ids, Mul, int64_t);
      SAME_DIMS_OP(last_ids_update, last_ids, last_ids, Add, int64_t);
    }
    TransCompute<DeviceContext, int64_t>(2, dev_ctx, tpath, path, {1, 0});
  }
};
}  // namespace operators
}  // namespace paddle
