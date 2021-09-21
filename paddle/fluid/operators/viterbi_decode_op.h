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
#include <type_traits>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/arg_min_max_op_base.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

#define CREATE_TENSOR(tensor, dtype, ...)             \
  LoDTensor tensor;                                   \
  tensor.Resize(framework::make_ddim({__VA_ARGS__})); \
  tensor.mutable_data<dtype>(ctx.GetPlace())

#define ELEMENT_BINARY_OP(lhs, rhs, output, functor_type, dtype)            \
  ElementwiseComputeEx<functor_type##Functor<dtype>, DeviceContext, dtype>( \
      ctx, &lhs, &rhs, -1, functor_type##Functor<dtype>(), &output)

#define ADD(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Add, dtype)

#define SUB(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Sub, dtype)

#define MUL(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Mul, dtype)

template <typename DeviceContext, typename T, size_t D, size_t R_D>
inline void MAX_FUNC(const framework::ExecutionContext& ctx,
                     const Tensor* input, Tensor* output,
                     const std::vector<int>& dims) {
  auto x =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  std::vector<int> dims_ref = dims;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    reduce_dim[i] = dims_ref[i];
  }
  DDim out_dims = output->dims();
  auto& place = *dev_ctx.eigen_device();
  if (D == 1) {
    auto out =
        framework::EigenScalar<T, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *output);
    out.device(place) = x.maximum(reduce_dim);
  } else {
    auto out =
        framework::EigenTensor<T, (D - R_D), Eigen::RowMajor,
                               Eigen::DenseIndex>::From(*output, out_dims);
    out.device(place) = x.maximum(reduce_dim);
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

    // Create a large int data buffer
    int buffer_size = batch_size * seq_len + batch_size * n_labels * seq_len +
                      7 * batch_size + 2;
    CREATE_TENSOR(int_buffer, int64_t, buffer_size);
    TensorBuffer int_tensor_buffer(int_buffer);

    // Create a large float data buffer
    buffer_size = seq_len * batch_size * n_labels + 5 * batch_size * n_labels +
                  2 * n_labels * n_labels + batch_size * n_labels * n_labels +
                  3 * batch_size;
    CREATE_TENSOR(float_buffer, T, buffer_size);
    TensorBuffer float_tensor_buffer(float_buffer);

    auto* length = ctx.Input<Tensor>("Length");
    Tensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);

    int64_t max_seq_len =
        *std::max_element(left_length.data<int64_t>(),
                          left_length.data<int64_t>() + left_length.numel());

    auto* scores = ctx.Output<Tensor>("Scores");
    scores->mutable_data<T>(curr_place);

    auto* path = ctx.Output<Tensor>("Path");
    path->Resize({batch_size, max_seq_len});
    path->mutable_data<int64_t>(curr_place);

    Tensor temp_path =
        int_tensor_buffer.GetBufferBlock({max_seq_len, batch_size});
    auto batch_path = Unbind(temp_path);
    for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
      it->Resize({batch_size});
    }

    Tensor inputs_t_exp =
        float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    std::vector<int> axis{1, 0, 2};
    TransCompute<DeviceContext, T>(axis.size(), dev_ctx, *input, &inputs_t_exp,
                                   axis);

    auto* transition = ctx.Input<Tensor>("Transition");
    Tensor trans_exp =
        float_tensor_buffer.GetBufferBlock({1, n_labels, n_labels});
    framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);

    Tensor alpha = float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    math::SetConstant<DeviceContext, T> float_functor;
    math::SetConstant<DeviceContext, int64_t> int_functor;

    std::vector<Tensor> historys;
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
    ArgMinMaxFunctor<DeviceContext, T, int64_t, 3, ArgMinMaxType::kArgMax>
        argmax3;

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
      ADD(logit0, start_trans_exp, alpha, T);
      ElementwiseComputeEx<EqualFunctor<T>, DeviceContext, int64_t, T>(
          ctx, &left_length, &one, -1, EqualFunctor<T>(), &float_mask);
      MUL(stop_trans_exp, float_mask, alpha_nxt, T);
      ADD(alpha, alpha_nxt, alpha, T);
    } else {
      alpha = logit0;
    }
    SUB(left_length, one, left_length, int64_t);
    for (int64_t i = 1; i < max_seq_len; ++i) {
      Tensor logit = inputs_t_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});

      ADD(alpha_exp, trans_exp, alpha_trn_sum, T);

      MAX_FUNC<DeviceContext, T, 3, 1>(ctx, &alpha_trn_sum, &alpha_max,
                                       std::vector<int>({1}));

      auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
      alpha_argmax_temp.Resize({batch_size, n_labels});
      argmax3(dev_ctx, static_cast<LoDTensor&>(alpha_trn_sum),
              reinterpret_cast<LoDTensor*>(&alpha_argmax_temp),
              alpha_trn_sum.dims(), 1, false);
      historys.push_back(alpha_argmax_temp);

      ADD(alpha_max, logit, alpha_nxt, T);

      alpha.Resize({batch_size, n_labels});

      // mask = paddle.cast((left_length > 0), dtype='float32')
      // alpha = mask * alpha_nxt + (1 - mask) * alpha
      ElementwiseComputeEx<GreaterThanFunctor<T>, DeviceContext, int64_t, T>(
          ctx, &left_length, &zero, -1, GreaterThanFunctor<T>(), &float_mask);
      // alpha_nxt = mask * alpha_nxt
      MUL(alpha_nxt, float_mask, alpha_nxt, T);
      // inv_mask = 1 - mask
      SUB(float_one, float_mask, float_mask, T);
      // alpha = (1 - mask) * alpha
      MUL(alpha, float_mask, alpha, T);
      // alpha += alpha_nxt
      ADD(alpha, alpha_nxt, alpha, T);

      if (with_start_stop_tag) {  // cost 10% time
        ElementwiseComputeEx<EqualFunctor<T>, DeviceContext, int64_t, T>(
            ctx, &left_length, &one, -1, EqualFunctor<T>(), &float_mask);
        // trans_exp: [1, n, n]
        // alpha += mask * trans_exp[:, self.stop_idx]
        MUL(stop_trans_exp, float_mask, alpha_nxt, T);
        ADD(alpha, alpha_nxt, alpha, T);
      }
      SUB(left_length, one, left_length, int64_t);
    }

    // scores, last_ids = alpha.max(1), alpha.argmax(1)
    MAX_FUNC<DeviceContext, T, 2, 1>(ctx, &alpha, scores,
                                     std::vector<int>({1}));
    ArgMinMaxFunctor<DeviceContext, T, int64_t, 2, ArgMinMaxType::kArgMax>
        argmax2;

    argmax2(dev_ctx, static_cast<LoDTensor&>(alpha),
            reinterpret_cast<LoDTensor*>(&last_ids), alpha.dims(), 1, false);

    // tag_mask = paddle.cast((left_length >= 0), 'int64')
    left_length.Resize({batch_size});
    ElementwiseComputeEx<GreaterEqualFunctor<int64_t>, DeviceContext, int64_t>(
        ctx, &left_length, &zero, -1, GreaterEqualFunctor<int64_t>(),
        &int_mask);

    // last_ids_update = last_ids * tag_mask
    int last_ids_index = 1;
    int actual_len = std::min(seq_len, static_cast<int>(max_seq_len));

    MUL(last_ids, int_mask, batch_path[actual_len - last_ids_index], int64_t);
    int64_t* batch_offset_ptr = batch_offset.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
      batch_offset_ptr[i] = i * n_labels;
    }

    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      ++last_ids_index;
      ADD(left_length, one, left_length, int64_t);
      ADD(batch_offset, last_ids, gather_idx, int64_t);
      // tag_mask = paddle.cast((left_length >= 0), 'int64')
      // last_ids_update = paddle.gather(hist.flatten(), gather_idx) * tag_mask
      Tensor& last_ids_update = batch_path[actual_len - last_ids_index];
      hist->Resize({batch_size * n_labels});
      CPUGather<int64_t, int64_t>(dev_ctx, *hist, gather_idx, &last_ids_update);
      ElementwiseComputeEx<GreaterEqualFunctor<int64_t>, DeviceContext,
                           int64_t>(ctx, &left_length, &zero, -1,
                                    GreaterEqualFunctor<int64_t>(), &int_mask);
      MUL(last_ids_update, int_mask, last_ids_update, int64_t);
      // tag_mask = 1 - tag_mask
      SUB(one, int_mask, int_mask, int64_t);
      // last_ids = last_ids_update + last_ids * (1 - tag_mask)
      MUL(last_ids, int_mask, last_ids, int64_t);
      ADD(last_ids_update, last_ids, last_ids, int64_t);
    }
    // transpose batch_path
    axis = {1, 0};
    TransCompute<DeviceContext, int64_t>(axis.size(), dev_ctx, temp_path, path,
                                         axis);
  }
};

}  // namespace operators
}  // namespace paddle
