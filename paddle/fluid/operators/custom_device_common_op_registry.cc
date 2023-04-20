/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/operators/collective/c_concat_op.h"
#include "paddle/fluid/operators/collective/c_identity_op.h"
#include "paddle/fluid/operators/load_combine_op.h"
#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

#define REGISTER_OP_CUSTOM_DEVICE_KERNEL(op_type, dev_type, ...)             \
  static paddle::framework::OpKernelRegistrar<phi::CustomPlace, __VA_ARGS__> \
      __op_custom_device_kernel_registrar_##op_type##_##__acosf##__(         \
          #op_type,                                                          \
          dev_type,                                                          \
          paddle::framework::OpKernelType::kDefaultCustomizedTypeValue);     \
  __op_custom_device_kernel_registrar_##op_type##_##__acosf##__.Touch();

#define REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL(                             \
    kernel_name, dev_type, layout, kernel_fn)                              \
  static phi::KernelRegistrar                                              \
      __reg_custom_device_phi_kernel_##kernel_name##_##backend##_##layout( \
          phi::RegType::INNER,                                             \
          #kernel_name,                                                    \
          dev_type,                                                        \
          DATA_LAYOUT(layout),                                             \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,      \
          [](const phi::KernelKey& kernel_key, phi::Kernel* kernel) {},    \
          PHI_KERNEL(kernel_fn),                                           \
          PHI_VARIADIC_KERNEL(kernel_fn))

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CConcatOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(rank,
                      0,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_concat must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

    auto& dev_ctx = ctx.template device_context<phi::CustomContext>();
    phi::DenseTensor temp_out;
    framework::DDim temp_out_dims = x->dims();
    temp_out_dims[0] *= nranks;
    temp_out.Resize(temp_out_dims);
    dev_ctx.template Alloc<T>(&temp_out);

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*x);
      out_tensor.push_back(temp_out);
      auto task = pg->AllGather(in_tensor, out_tensor);
      task->Wait();
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "CustomDevice c_concat only support ProcessGroup"));
    }
    std::vector<phi::DenseTensor> inputs;
    int axis = x->dims().size() - 1;
    auto out_dims = x->dims();
    out_dims[out_dims.size() - 1] *= nranks;
    int rows_per_tensor = x->dims()[0];
    int offset = 0;
    for (int i = 0; i < nranks; i++) {
      phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
      inputs.emplace_back(temp);
      offset += rows_per_tensor;
    }

    out->Resize(out_dims);
    std::vector<paddle::Tensor> inputs_t(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      auto t = std::make_shared<phi::DenseTensor>();
      t->ShareDataWith(inputs[i]);
      inputs_t[i].set_impl(t);
    }
    auto output = paddle::experimental::concat(inputs_t, axis);
    out->ShareDataWith(
        *reinterpret_cast<phi::DenseTensor*>(output.impl().get()));
  }
};

template <typename DeviceContext, typename T>
class CSplitOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");

    PADDLE_ENFORCE_GE(rank,
                      0,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

    auto dims = x->dims();
    auto dims_size = dims.size();

    dims[dims_size - 1] /= nranks;
    out->Resize(dims);
    std::vector<int64_t> split_list(nranks, dims[dims_size - 1]);
    int axis = dims_size - 1;

    auto x_tmp = std::make_shared<phi::DenseTensor>();
    x_tmp->ShareDataWith(*x);
    paddle::Tensor x_tensor(x_tmp);
    auto outputs = paddle::experimental::split(x_tensor, split_list, axis);
    out->ShareDataWith(
        *reinterpret_cast<phi::DenseTensor*>(outputs[rank].impl().get()));
  }
};

template <typename DeviceContext, typename T>
class CEmbeddingOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto* table_t = ctx.Input<phi::DenseTensor>("W");
    auto* output_t = ctx.Output<phi::DenseTensor>("Out");
    auto out_dims = output_t->dims();
    auto start_index = ctx.Attr<int64_t>("start_index");

    auto K = ids_t->numel();
    auto N = table_t->dims()[0];
    auto D = table_t->dims()[1];
    auto index_type = ids_t->dtype();
    if (index_type == phi::DataType::INT32 ||
        index_type == phi::DataType::INT64) {
      auto x_tmp = std::make_shared<phi::DenseTensor>();
      x_tmp->ShareDataWith(*ids_t).Resize({K});
      auto w_tmp = std::make_shared<phi::DenseTensor>();
      w_tmp->ShareDataWith(*table_t).Resize({N, D});
      paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp);
      auto start_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
      auto end_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
      auto ids_mask_tensor = paddle::experimental::logical_and(
          x_tensor.greater_equal(start_index_tensor),
          x_tensor.less_than(end_index_tensor));
      auto ids_tensor = (x_tensor - start_index_tensor)
                            .multiply(paddle::experimental::cast(
                                ids_mask_tensor, x_tensor.dtype()));
      auto out_tensor =
          paddle::experimental::reshape(
              paddle::experimental::cast(ids_mask_tensor, w_tensor.dtype()),
              {K, 1})
              .multiply(paddle::experimental::reshape(
                  paddle::experimental::embedding(
                      ids_tensor, w_tensor, -1, false),
                  {K, D}));
      output_t
          ->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(out_tensor.impl().get()))
          .Resize(out_dims);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "CustomDevice c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename DeviceContext, typename T>
class CEmbeddingGradOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto start_index = ctx.Attr<int64_t>("start_index");
    auto ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto d_output_t =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto table_t = ctx.Input<phi::DenseTensor>("W");
    auto table_grad_t =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("W"));
    table_grad_t->Resize(table_t->dims());
    auto& dev_ctx = ctx.template device_context<phi::CustomContext>();

    auto K = ids_t->numel();
    auto N = table_t->dims()[0];
    auto D = table_t->dims()[1];
    const auto& index_type = ids_t->dtype();
    if (index_type == phi::DataType::INT32 ||
        index_type == phi::DataType::INT64) {
      auto x_tmp = std::make_shared<phi::DenseTensor>();
      x_tmp->ShareDataWith(*ids_t).Resize({K});
      auto w_tmp = std::make_shared<phi::DenseTensor>();
      w_tmp->set_meta(table_t->meta());
      dev_ctx.Alloc(w_tmp.get(), w_tmp->dtype());
      auto out_grad_tmp = std::make_shared<phi::DenseTensor>();
      out_grad_tmp->ShareDataWith(*d_output_t).Resize({K, D});
      paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp),
          out_grad_tensor(out_grad_tmp);
      auto start_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
      auto end_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
      auto ids_mask_tensor = paddle::experimental::logical_and(
          x_tensor.greater_equal(start_index_tensor),
          x_tensor.less_equal(end_index_tensor));
      auto real_ids_tensor = (x_tensor - start_index_tensor)
                                 .multiply(paddle::experimental::cast(
                                     ids_mask_tensor, x_tensor.dtype()));
      auto out_grad_tensor_mul_mask =
          paddle::experimental::reshape(out_grad_tensor, {K, D})
              .multiply(paddle::experimental::reshape(
                  paddle::experimental::cast(ids_mask_tensor, table_t->dtype()),
                  {K, 1}));
      paddle::Tensor table_grad_tensor;
      paddle::experimental::embedding_grad(real_ids_tensor,
                                           w_tensor,
                                           out_grad_tensor_mul_mask,
                                           -1,
                                           false,
                                           &table_grad_tensor);
      table_grad_t->ShareDataWith(
          *reinterpret_cast<phi::DenseTensor*>(table_grad_tensor.impl().get()));
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "CustomDevice c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyOpCustomDeviceKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
      const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
      phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
      phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");
      auto softmax_dims = softmax->dims();
      auto loss_dims = loss->dims();

      const int64_t ignore_index = ctx.Attr<int64_t>("ignore_index");
      PADDLE_ENFORCE_LT(ignore_index,
                        0,
                        platform::errors::InvalidArgument(
                            "When SoftmaxWithCrossEntropy run on CustomDevice, "
                            "ignore_index should be <=0, however it's %ld",
                            ignore_index));
      const int rid = ctx.Attr<int>("ring_id");
      const int rank = ctx.Attr<int>("rank");

      distributed::ProcessGroup* pg = map->get(rid);
      distributed::AllreduceOptions opts;

      // allocate memory on device.
      const auto& logits_dims = logits->dims();

      const int axis = logits_dims.size() - 1;
      const int N = phi::funcs::SizeToAxis(axis, logits_dims);
      const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

      auto logits_2d = std::make_shared<phi::DenseTensor>();
      auto labels_1d = std::make_shared<phi::DenseTensor>();
      logits_2d->ShareDataWith(*logits).Resize({N, D});
      labels_1d->ShareDataWith(*labels).Resize({N});
      paddle::Tensor logits_2d_tensor(logits_2d), labels_1d_tensor(labels_1d);

      // step 1, obtain logit_max
      auto logits_2d_max_tensor = logits_2d_tensor.max({1}, true);
      std::vector<phi::DenseTensor> in_out;
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          logits_2d_max_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::MAX;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      // step 2, obtain logit - logit_max
      auto logits_2d_sub_max = paddle::experimental::clip(
          logits_2d_tensor - logits_2d_max_tensor, -64., 0.);

      // step 3, obtain predict target
      const int start_index = rank * D;
      auto start_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          start_index,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());
      auto end_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          start_index + D,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());
      auto labels_1d_mask = paddle::experimental::logical_and(
          labels_1d_tensor.greater_equal(start_index_tensor),
          labels_1d_tensor.less_than(end_index_tensor));
      auto real_label_tensor =
          (labels_1d_tensor - start_index_tensor)
              .multiply(paddle::experimental::cast(labels_1d_mask,
                                                   labels_1d_tensor.dtype()));

      auto predicted_logits_tensor =
          logits_2d_sub_max
              .multiply(paddle::experimental::cast(
                  paddle::experimental::one_hot(real_label_tensor, D),
                  logits_2d_sub_max.dtype()))
              .sum({1}, logits_2d_sub_max.dtype(), false)
              .multiply(paddle::experimental::cast(labels_1d_mask,
                                                   logits_2d_sub_max.dtype()));

      in_out.clear();
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          predicted_logits_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::SUM;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      // step 4, obtain exp(logit)
      auto softmax_2d_tensor = logits_2d_sub_max.exp();

      // step 5, obtain sum_exp_logits
      auto sum_exp_logits_tensor =
          softmax_2d_tensor.sum({1}, softmax_2d_tensor.dtype(), false);

      in_out.clear();
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          sum_exp_logits_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::SUM;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      auto softmax_out = softmax_2d_tensor.divide(
          paddle::experimental::reshape(sum_exp_logits_tensor, {N, 1}));
      auto labels_1d_not_equal_ignore = labels_1d_tensor.not_equal(
          paddle::experimental::full_like(labels_1d_tensor,
                                          ignore_index,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place()));
      auto loss_out =
          (sum_exp_logits_tensor.log() - predicted_logits_tensor)
              .multiply(paddle::experimental::cast(
                  labels_1d_not_equal_ignore, sum_exp_logits_tensor.dtype()));
      softmax
          ->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(softmax_out.impl().get()))
          .Resize(softmax_dims);
      loss->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(loss_out.impl().get()))
          .Resize(loss_dims);
    } else {
      PADDLE_THROW(
          phi::errors::Unavailable("CustomDevice c_softmax_with_cross_entropy "
                                   "only support ProcessGroup"));
    }
  }
};

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyGradCustomDeviceKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* loss_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    const phi::DenseTensor* softmax =
        context.Input<phi::DenseTensor>("Softmax");
    phi::DenseTensor* logit_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));

    const int64_t ignore_index = context.Attr<int64_t>("ignore_index");
    const int rank = context.Attr<int>("rank");
    if (logit_grad != softmax) {
      framework::TensorCopy(
          *softmax, context.GetPlace(), context.device_context(), logit_grad);
    }
    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, sofrmax_dims);
    const int D = phi::funcs::SizeFromAxis(axis, sofrmax_dims);
    const auto& label_type = labels->dtype();

    if (label_type == phi::DataType::INT32 ||
        label_type == phi::DataType::INT64) {
      auto logit_grad_t = std::make_shared<phi::DenseTensor>();
      logit_grad_t->ShareDataWith(*logit_grad).Resize({N, D});
      auto loss_grad_t = std::make_shared<phi::DenseTensor>();
      loss_grad_t->ShareDataWith(*loss_grad).Resize({N});
      auto labels_1d = std::make_shared<phi::DenseTensor>();
      labels_1d->ShareDataWith(*labels).Resize({N});
      paddle::Tensor logits_grad_tensor(logit_grad_t),
          loss_grad_tensor(loss_grad_t), labels_1d_tensor(labels_1d);

      auto labels_1d_not_equal_ignore = paddle::experimental::reshape(
          paddle::experimental::not_equal(
              labels_1d_tensor,
              paddle::experimental::full_like(labels_1d_tensor,
                                              ignore_index,
                                              labels_1d_tensor.dtype(),
                                              labels_1d_tensor.place())),
          {N, 1});
      auto start_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          rank * D,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());

      auto logits_grad_out_tensor1 = paddle::experimental::subtract(
          paddle::experimental::multiply(
              logits_grad_tensor,
              paddle::experimental::cast(labels_1d_not_equal_ignore,
                                         logits_grad_tensor.dtype())),
          paddle::experimental::cast(
              paddle::experimental::one_hot(
                  paddle::experimental::subtract(labels_1d_tensor,
                                                 start_index_tensor),
                  D),
              logits_grad_tensor.dtype()));

      auto logits_grad_out_tensor2 = paddle::experimental::multiply(
          logits_grad_out_tensor1,
          paddle::experimental::reshape(loss_grad_tensor, {N, 1}));
      logit_grad
          ->ShareDataWith(*reinterpret_cast<phi::DenseTensor*>(
              logits_grad_out_tensor2.impl().get()))
          .Resize(sofrmax_dims);
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "CustomDevice c_softmax_with_cross_entropy_grad "
          "label_type only support int32/int64"));
    }
  }
};

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out);

void RegisterCustomDeviceCommonKernel(const std::string& dev_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto device_type = dev_type.c_str();
  /* see [Why use single type kernel] */
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      run_program,
      device_type,
      paddle::operators::
          RunProgramOpKernel<float, paddle::platform::CustomDeviceContext>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      run_program_grad,
      device_type,
      paddle::operators ::
          RunProgramGradOpKernel<float, paddle::platform::CustomDeviceContext>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      save_combine,
      device_type,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, float>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, double>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, int>,
      paddle::operators ::
          SaveCombineOpKernel<paddle::platform::CustomDeviceContext, int64_t>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      load_combine,
      device_type,
      paddle::operators::
          LoadCombineOpKernel<float, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          LoadCombineOpKernel<double, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          LoadCombineOpKernel<int, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          LoadCombineOpKernel<int8_t, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          LoadCombineOpKernel<int64_t, paddle::platform::CustomDeviceContext>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_concat,
      device_type,
      paddle::operators::CConcatOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>,
      paddle::operators::CConcatOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          paddle::platform::float16>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_split,
      device_type,
      paddle::operators::CSplitOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>,
      paddle::operators::CSplitOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          int>,
      paddle::operators::CSplitOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          paddle::platform::float16>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_embedding,
      device_type,
      paddle::operators::CEmbeddingOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>);
  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_embedding_grad,
      device_type,
      paddle::operators::CEmbeddingGradOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>);

  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_softmax_with_cross_entropy,
      device_type,
      paddle::operators::CSoftmaxWithCrossEntropyOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>,
      paddle::operators::CSoftmaxWithCrossEntropyOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          double>,
      paddle::operators::CSoftmaxWithCrossEntropyOpCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          paddle::platform::float16>) {}

  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_softmax_with_cross_entropy_grad,
      device_type,
      paddle::operators::CSoftmaxWithCrossEntropyGradCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          float>,
      paddle::operators::CSoftmaxWithCrossEntropyGradCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          double>,
      paddle::operators::CSoftmaxWithCrossEntropyGradCustomDeviceKernel<
          paddle::platform::CustomDeviceContext,
          paddle::platform::float16>) {}

  REGISTER_OP_CUSTOM_DEVICE_KERNEL(
      c_identity,
      device_type,
      paddle::operators::
          CIdentityOpKernel<float, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          CIdentityOpKernel<double, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          CIdentityOpKernel<int, paddle::platform::CustomDeviceContext>,
      paddle::operators::
          CIdentityOpKernel<int64_t, paddle::platform::CustomDeviceContext>,
      paddle::operators::CIdentityOpKernel<
          paddle::platform::float16,
          paddle::platform::CustomDeviceContext>) {}

#endif
}

}  // namespace operators
}  // namespace paddle

#undef REGISTER_OP_CUSTOM_DEVICE_KERNEL
#undef REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL
