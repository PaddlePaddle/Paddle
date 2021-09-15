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
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

#define CREATE_TENSOR(tensor, ...)                    \
  LoDTensor tensor;                                   \
  tensor.Resize(framework::make_ddim({__VA_ARGS__})); \
  tensor.mutable_data<T>(ctx.GetPlace())

#define ELE_MAX(input, output, dims)                                          \
  auto cast_out_dtype =                                                       \
      static_cast<framework::proto::VarType::Type>(output.type());            \
  framework::VisitDataType(cast_out_dtype,                                    \
                           ReduceKernelFunctor<DeviceContext, T, MaxFunctor>( \
                               &input, &output, dims, false, false, ctx));

#define ELEMENT_BINARY_OP(lhs, rhs, output, Functor)                      \
  ElementwiseComputeEx<Functor<T>, DeviceContext, T>(ctx, &lhs, &rhs, -1, \
                                                     Functor<T>(), &output)

// output = lhs + rhs
#define ADD(lhs, rhs, output) ELEMENT_BINARY_OP(lhs, rhs, output, AddFunctor)

#define SUB(lhs, rhs, output) ELEMENT_BINARY_OP(lhs, rhs, output, SubFunctor)

#define INVERSE_SUB(lhs, rhs, output) \
  ELEMENT_BINARY_OP(lhs, rhs, output, InverseSubFunctor)

#define MUL(lhs, rhs, output) ELEMENT_BINARY_OP(lhs, rhs, output, MulFunctor)

#define GET_CAST_MASK(lhs, rhs, mask, compare_functor, dtype)           \
  ElementwiseComputeEx<compare_functor<int64_t>, DeviceContext, bool>(  \
      ctx, &lhs, &rhs, -1, compare_functor<int64_t>(), &mask);          \
  CastOpFunctor<DeviceContext, bool> cast_functor(&tag_mask, &tag_mask, \
                                                  dev_ctx);             \
  cast_functor.template apply<dtype>()

template <typename DeviceContext, typename T>
class ViterbiDecodeCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool with_start_stop_tag = ctx.Attr<bool>("with_start_stop_tag");

    auto* input = ctx.Input<Tensor>("Input");
    auto batch_size = static_cast<int>(input->dims()[0]);
    auto seq_len = static_cast<int>(input->dims()[1]);
    auto n_labels = static_cast<int>(input->dims()[2]);

    auto* scores = ctx.Output<Tensor>("Scores");
    auto* path = ctx.Output<Tensor>("Path");

    CREATE_TENSOR(temp_path, seq_len, batch_size);
    auto batch_path = Unbind(temp_path);
    for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
      it->Resize({batch_size});
    }

    std::vector<int> axis{seq_len, batch_size, n_labels};
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto curr_place = ctx.GetPlace();

    CREATE_TENSOR(inputs_t_exp, seq_len, batch_size, n_labels);
    TransCompute<DeviceContext, T>(axis.size(), dev_ctx, *input, &inputs_t_exp,
                                   axis);

    auto* transition = ctx.Input<Tensor>("Transition");
    CREATE_TENSOR(trans_exp, 1, n_labels, n_labels);
    framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);

    auto* length = ctx.Input<Tensor>("Length");
    CREATE_TENSOR(left_length, batch_size, 1);
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);

    int64_t max_seq_len =
        *std::max_element(left_length.data<int64_t>(),
                          left_length.data<int64_t>() + left_length.numel());

    CREATE_TENSOR(alpha, batch_size, n_labels);
    math::SetConstant<platform::CPUDeviceContext, T> functor;

    if (with_start_stop_tag) {
      CREATE_TENSOR(initial_alpha, batch_size, n_labels - 1);
      functor(dev_ctx, &initial_alpha, static_cast<T>(-10000.0));
      CREATE_TENSOR(alpha_start, batch_size, 1);
      functor(dev_ctx, &alpha_start, static_cast<T>(0.0));

      math::ConcatFunctor<platform::CPUDeviceContext, T> concat_functor;
      concat_functor(dev_ctx, {initial_alpha, alpha_start}, 1, &alpha);
    } else {
      functor(dev_ctx, &alpha, static_cast<T>(0.0));
    }

    std::vector<Tensor> historys;
    CREATE_TENSOR(zero, 1);
    CREATE_TENSOR(one, 1);
    CREATE_TENSOR(alpha_trn_sum, batch_size, n_labels, n_labels);
    CREATE_TENSOR(alpha_max, batch_size, n_labels);
    CREATE_TENSOR(alpha_argmax, batch_size, n_labels);
    CREATE_TENSOR(alpha_nxt, batch_size, n_labels);
    CREATE_TENSOR(mask, batch_size, 1);
    CREATE_TENSOR(inv_mask, batch_size, 1);
    CREATE_TENSOR(tag_mask, batch_size);
    CREATE_TENSOR(alpha_temp, batch_size, 1);
    CREATE_TENSOR(stop_trans_exp, 1, n_labels, 1);
    CREATE_TENSOR(start_trans_exp, 1, n_labels, 1);
    CREATE_TENSOR(rest_trans_exp, 1, n_labels, n_labels - 2);
    CREATE_TENSOR(last_ids, batch_size);
    CREATE_TENSOR(batch_offset, batch_size);
    CREATE_TENSOR(gather_idx, batch_size);

    for (int64_t i = 0; i < max_seq_len; ++i) {
      Tensor logit = inputs_t_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      if (i == 0 && !with_start_stop_tag) {
        framework::TensorCopy(logit, curr_place, dev_ctx, &alpha);
        SUB(left_length, one, left_length);
      }
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});

      ADD(alpha_exp, trans_exp, alpha_trn_sum);

      ELE_MAX(alpha_trn_sum, alpha_max, {1});

      if (i >= 1) {
        ArgMinMaxFunctor<DeviceContext, T, int64_t, 3, ArgMinMaxType::kArgMax>
            argmax;
        argmax(dev_ctx, alpha_trn_sum, &alpha_argmax, alpha_trn_sum.dims(), 1,
               false);
        historys.push_back(alpha_argmax);
      }

      ADD(alpha_max, logit, alpha_nxt);

      alpha.Resize({batch_size, n_labels});

      // mask = paddle.cast((left_length > 0), dtype='float32')
      // alpha = mask * alpha_nxt + (1 - mask) * alpha
      GET_CAST_MASK(left_length, zero, mask, GreaterThanFunctor, T);

      // inv_mask = 1 - mask
      INVERSE_SUB(one, mask, inv_mask);
      // alpha = (1 - mask) * alpha
      MUL(alpha, inv_mask, alpha);
      // alpha_temp = mask * alpha_nxt
      MUL(alpha_nxt, mask, alpha_temp);
      // alpha += alpha_temp
      ADD(alpha, alpha_temp, alpha);

      if (with_start_stop_tag) {
        GET_CAST_MASK(left_length, one, mask, EqualFunctor, T);
        // trans_exp: [1, n, n]
        // alpha += mask * trans_exp[:, self.stop_idx]
        std::vector<const Tensor*> shape_refer{&rest_trans_exp, &stop_trans_exp,
                                               &start_trans_exp};
        std::vector<Tensor*> outputs{&rest_trans_exp, &stop_trans_exp,
                                     &start_trans_exp};
        math::SplitFunctor<DeviceContext, T> functor;
        functor(dev_ctx, trans_exp, shape_refer, 2, &outputs);

        stop_trans_exp.Resize({1, n_labels});
        MUL(stop_trans_exp, mask, alpha_temp);
        stop_trans_exp.Resize({1, n_labels, 1});
        ADD(alpha, alpha_temp, alpha);
      }
      // left_length = left_length - 1
      SUB(left_length, one, left_length);
    }

    // scores, last_ids = alpha.max(1), alpha.argmax(1)
    ELE_MAX(alpha, (*scores), {1});
    ArgMinMaxFunctor<DeviceContext, T, int64_t, 2, ArgMinMaxType::kArgMax>
        argmax;
    argmax(dev_ctx, alpha, &last_ids, alpha.dims(), 1, false);

    // tag_mask = paddle.cast((left_length >= 0), 'int64')
    left_length.Resize({batch_size});
    GET_CAST_MASK(left_length, zero, tag_mask, GreaterEqualFunctor, int64_t);

    // last_ids_update = last_ids * tag_mask
    int last_ids_index = 1;
    MUL(last_ids, tag_mask, batch_path[seq_len - last_ids_index]);

    int64_t* batch_offset_ptr = batch_offset.data<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
      batch_offset_ptr[i] = i * n_labels;
    }

    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      --last_ids_index;
      ADD(left_length, one, left_length);
      ADD(batch_offset, last_ids, gather_idx);
      // tag_mask = paddle.cast((left_length >= 0), 'int64')
      // last_ids_update = paddle.gather(hist.flatten(), gather_idx) * tag_mask
      hist->Resize({batch_size * n_labels});
      CPUGather<T, int64_t>(dev_ctx, *hist, gather_idx,
                            &batch_path[seq_len - last_ids_index]);
      GET_CAST_MASK(left_length, zero, tag_mask, GreaterEqualFunctor, int64_t);
      MUL(batch_path[seq_len - last_ids_index], tag_mask,
          batch_path[seq_len - last_ids_index]);

      // tag_mask = 1 - tag_mask
      INVERSE_SUB(one, tag_mask, tag_mask);
      // last_ids = last_ids_update + last_ids * (1 - tag_mask)
      MUL(last_ids, tag_mask, last_ids);
      ADD(batch_path[seq_len - last_ids_index], last_ids, last_ids);
    }
    // transpose batch_path
    TransCompute<DeviceContext, T>(2, dev_ctx, temp_path, path,
                                   {batch_size, seq_len});
  }
};

}  // namespace operators
}  // namespace paddle
