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

template <typename DeviceContext, typename T, typename IndType>
struct Argmax {
  void operator()(const framework::ExecutionContext& ctx, const Tensor& input,
                  Tensor* out_idx, Tensor* out, int axis) {
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

template <typename DeviceContext>
struct ARange {
  void operator()(const DeviceContext& dev_ctx, int64_t* data, int end,
                  int64_t scale) {
    for (int i = 0; i < end; ++i) {
      data[i] = i * scale;
    }
  }
};

template <typename DeviceContext, typename T>
struct GetMaxValue {
  void operator()(const DeviceContext& dev_ctx, const Tensor& input,
                  T* max_value) {
    auto input_ptr = input.data<T>();
    auto num = input.numel();
    *max_value = *std::max_element(input_ptr, input_ptr + num);
  }
};

template <typename DeviceContext, typename T, typename IndexT = int>
struct Gather {
  void operator()(const DeviceContext& ctx, const Tensor& src,
                  const Tensor& index, Tensor* output) {
    CPUGather<T, IndexT>(ctx, src, index, output);
  }
};

template <typename T, typename Functor, typename OutT = T>
void SameDimsBinaryOP(const Tensor& lhs, const Tensor& rhs, Tensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  OutT* out_ptr = out->data<OutT>();
  Functor functor;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < out->numel(); ++i) {
    out_ptr[i] = functor(lhs_ptr[i], rhs_ptr[i]);
  }
}

template <typename DeviceContext, template <typename T> typename CompareFunctor,
          typename T>
struct GetMask {
  void operator()(const framework::ExecutionContext& ctx, const Tensor& lhs,
                  const Tensor& rhs, Tensor* mask) {
    SameDimsBinaryOP<int64_t, CompareFunctor<int64_t>, T>(lhs, rhs, mask);
  }
};

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
    *lhs_idx =
        pten::GetElementwiseIndex(lhs_dims.data(), out_dims_size, index_array);
    *rhs_idx =
        pten::GetElementwiseIndex(rhs_dims.data(), out_dims_size, index_array);
    pten::UpdateElementwiseIndexArray(output_dims.data(), out_dims_size,
                                      index_array);
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

template <typename DeviceContext, template <typename T> typename BinaryFunctor,
          typename T>
struct BinaryOperation {
  void operator()(const DeviceContext& dev_ctx, const Tensor& lhs,
                  const Tensor& rhs, Tensor* output) {
    if (lhs.dims() == rhs.dims()) {
      SameDimsBinaryOP<T, BinaryFunctor<T>>(lhs, rhs, output);
    } else {
      bool is_multi_threads = false;
#ifdef PADDLE_WITH_MKLML
      if (omp_get_max_threads() > 1) {
        is_multi_threads = true;
      }
#endif
      if (is_multi_threads) {
        SimpleBroadcastBinaryOP<T, BinaryFunctor<T>, true>(lhs, rhs, output);
      } else {
        SimpleBroadcastBinaryOP<T, BinaryFunctor<T>, false>(lhs, rhs, output);
      }
    }
  }
};

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
    bool include_bos_eos_tag = ctx.Attr<bool>("include_bos_eos_tag");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto curr_place = ctx.GetPlace();
    auto* input = ctx.Input<Tensor>("Input");
    auto batch_size = static_cast<int>(input->dims()[0]);
    auto seq_len = static_cast<int>(input->dims()[1]);
    auto n_labels = static_cast<int>(input->dims()[2]);
    math::SetConstant<DeviceContext, T> float_functor;
    math::SetConstant<DeviceContext, int64_t> int_functor;
    std::vector<Tensor> historys;
    // We create tensor buffer in order to avoid allocating memory frequently
    // 10 means allocate 10*batch_size bytes memory, such as int_mask, zero...
    int buffer_size = batch_size * (n_labels + 1) * seq_len + 10 * batch_size;
    LoDTensor int_buffer;
    int_buffer.Resize(framework::make_ddim({buffer_size}));
    int_buffer.mutable_data<int64_t>(ctx.GetPlace());
    TensorBuffer int_tensor_buffer(int_buffer);
    // create float tensor buffer
    // 10 means allocate 10*batch_size*n_labels bytes, such as alpha, alpha_max
    buffer_size = batch_size * (seq_len + 10) * n_labels +
                  (batch_size + 2) * n_labels * n_labels;
    LoDTensor float_buffer;
    float_buffer.Resize(framework::make_ddim({buffer_size}));
    float_buffer.mutable_data<T>(ctx.GetPlace());
    TensorBuffer float_tensor_buffer(float_buffer);
    auto* length = ctx.Input<Tensor>("Length");
    Tensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);
    int64_t max_seq_len = 0;
    GetMaxValue<DeviceContext, int64_t> get_max_value;
    get_max_value(dev_ctx, left_length, &max_seq_len);

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
    Tensor input_exp =
        float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    TransCompute<DeviceContext, T>(3, dev_ctx, *input, &input_exp, {1, 0, 2});
    auto* transition = ctx.Input<Tensor>("Transition");
    Tensor trans_exp = float_tensor_buffer.GetBufferBlock({n_labels, n_labels});
    framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);
    trans_exp.Resize({1, n_labels, n_labels});
    Tensor alpha = float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor zero = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    int_functor(dev_ctx, &zero, 0);
    Tensor one = int_tensor_buffer.GetBufferBlock({batch_size, 1});
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
    Tensor zero_len_mask = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor float_mask = float_tensor_buffer.GetBufferBlock({batch_size, 1});
    Tensor stop_trans = float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor start_trans = float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor rest_trans =
        float_tensor_buffer.GetBufferBlock({1, n_labels - 2, n_labels});
    Tensor last_ids = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor last_ids_tmp = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor batch_offset = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor gather_idx = int_tensor_buffer.GetBufferBlock({batch_size});
    std::vector<const Tensor*> shape{&rest_trans, &stop_trans, &start_trans};
    std::vector<Tensor*> outputs{&rest_trans, &stop_trans, &start_trans};
    math::SplitFunctor<DeviceContext, T> split_functor;
    split_functor(dev_ctx, trans_exp, shape, 1, &outputs);
    stop_trans.Resize({1, n_labels});
    start_trans.Resize({1, n_labels});
    auto logit0 = input_exp.Slice(0, 1);
    logit0.Resize({batch_size, n_labels});
    BinaryOperation<DeviceContext, AddFunctor, T> AddFloat;
    BinaryOperation<DeviceContext, AddFunctor, int64_t> AddInt;
    BinaryOperation<DeviceContext, MulFunctor, T> MulFloat;
    BinaryOperation<DeviceContext, MulFunctor, int64_t> MulInt;
    BinaryOperation<DeviceContext, SubFunctor, T> SubFloat;
    BinaryOperation<DeviceContext, SubFunctor, int64_t> SubInt;
    if (include_bos_eos_tag) {
      AddFloat(dev_ctx, logit0, start_trans, &alpha);
      GetMask<DeviceContext, EqualFunctor, T>()(ctx, left_length, one,
                                                &float_mask);
      MulFloat(dev_ctx, stop_trans, float_mask, &alpha_nxt);
      AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
    } else {
      alpha = logit0;
    }
    SubInt(dev_ctx, left_length, one, &left_length);
    Argmax<DeviceContext, T, int64_t> argmax;
    for (int64_t i = 1; i < max_seq_len; ++i) {
      Tensor logit = input_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});
      AddFloat(dev_ctx, alpha_exp, trans_exp, &alpha_trn_sum);
      auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
      alpha_argmax_temp.Resize({batch_size, n_labels});
      argmax(ctx, alpha_trn_sum, &alpha_argmax_temp, &alpha_max, 1);
      historys.emplace_back(alpha_argmax_temp);
      AddFloat(dev_ctx, alpha_max, logit, &alpha_nxt);
      alpha.Resize({batch_size, n_labels});
      // mask = paddle.cast((left_length > 0), dtype='float32')
      // alpha = mask * alpha_nxt + (1 - mask) * alpha
      GetMask<DeviceContext, GreaterThanFunctor, T>()(ctx, left_length, zero,
                                                      &float_mask);
      // alpha_nxt = mask * alpha_nxt
      MulFloat(dev_ctx, alpha_nxt, float_mask, &alpha_nxt);
      // inv_mask = 1 - mask
      SubFloat(dev_ctx, float_one, float_mask, &float_mask);
      // alpha = (1 - mask) * alpha
      MulFloat(dev_ctx, alpha, float_mask, &alpha);
      // alpha += alpha_nxt
      AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
      if (include_bos_eos_tag) {
        GetMask<DeviceContext, EqualFunctor, T>()(ctx, left_length, one,
                                                  &float_mask);
        // alpha += mask * trans_exp[:, self.stop_idx]
        MulFloat(dev_ctx, stop_trans, float_mask, &alpha_nxt);
        AddFloat(dev_ctx, alpha, alpha_nxt, &alpha);
      }
      SubInt(dev_ctx, left_length, one, &left_length);
    }
    argmax(ctx, alpha, &last_ids, scores, 1);
    left_length.Resize({batch_size});
    GetMask<DeviceContext, GreaterEqualFunctor, int64_t>()(ctx, left_length,
                                                           zero, &int_mask);
    // last_ids_update = last_ids * tag_mask
    int last_ids_index = 1;
    int actual_len = (std::min)(seq_len, static_cast<int>(max_seq_len));
    MulInt(dev_ctx, last_ids, int_mask,
           &batch_path[actual_len - last_ids_index]);
    // The algorithm below can refer to
    // https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/layers/crf.py#L438
    ARange<DeviceContext> arange;
    arange(dev_ctx, batch_offset.data<int64_t>(), batch_size, n_labels);
    Gather<DeviceContext, int64_t, int64_t> gather;
    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      ++last_ids_index;
      AddInt(dev_ctx, left_length, one, &left_length);
      AddInt(dev_ctx, batch_offset, last_ids, &gather_idx);
      Tensor& last_ids_update = batch_path[actual_len - last_ids_index];
      hist->Resize({batch_size * n_labels});
      gather(dev_ctx, *hist, gather_idx, &last_ids_update);
      GetMask<DeviceContext, GreaterThanFunctor, int64_t>()(ctx, left_length,
                                                            zero, &int_mask);
      MulInt(dev_ctx, last_ids_update, int_mask, &last_ids_update);
      GetMask<DeviceContext, EqualFunctor, int64_t>()(ctx, left_length, zero,
                                                      &zero_len_mask);
      MulInt(dev_ctx, last_ids, zero_len_mask, &last_ids_tmp);
      SubInt(dev_ctx, one, zero_len_mask, &zero_len_mask);
      MulInt(dev_ctx, last_ids_update, zero_len_mask, &last_ids_update);
      AddInt(dev_ctx, last_ids_update, last_ids_tmp, &last_ids_update);
      GetMask<DeviceContext, LessThanFunctor, int64_t>()(ctx, left_length, zero,
                                                         &int_mask);
      MulInt(dev_ctx, last_ids, int_mask, &last_ids);
      AddInt(dev_ctx, last_ids_update, last_ids, &last_ids);
    }
    TransCompute<DeviceContext, int64_t>(2, dev_ctx, tpath, path, {1, 0});
  }
};
}  // namespace operators
}  // namespace paddle
