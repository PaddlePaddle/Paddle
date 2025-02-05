// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
#include "paddle/phi/kernels/xpu/reduce.h"
#include "paddle/utils/string/string_helper.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename Context, typename T>
struct CSoftmaxWithCrossEntropyFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& logits,
                  const DenseTensor& label,
                  int64_t ignore_index,
                  int rank,
                  int nranks,
                  DenseTensor* softmax,
                  DenseTensor* loss);
};

template <typename T, typename Context>
void CSoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                                    const DenseTensor& logits,
                                    const DenseTensor& label,
                                    int64_t ignore_index,
                                    int rank,
                                    int nranks,
                                    DenseTensor* softmax,
                                    DenseTensor* loss) {
  CSoftmaxWithCrossEntropyFunctor<Context, T> functor_;
  functor_(dev_ctx, logits, label, ignore_index, rank, nranks, softmax, loss);
}

template <typename T>
void FixLossAccordingToIgnoreIndex(const phi::XPUContext& dev_ctx,
                                   const phi::DenseTensor* labels,
                                   const phi::DenseTensor* predicted_logits,
                                   phi::DenseTensor* loss,
                                   const int64_t N,
                                   const int64_t ignore_index) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  // 先准备一个全0的tensor
  phi::DenseTensor zeros_constant;
  zeros_constant.Resize({N, 1});
  dev_ctx.template Alloc<T>(&zeros_constant);

  int ret = xpu::constant<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<XPUType*>(zeros_constant.data<T>()),
      N,
      0.0);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");

  // 准备一个bool类型的tensor，用来标记每一个loss要不要刷0
  phi::DenseTensor bool_tensor_for_mask_label;
  bool_tensor_for_mask_label.Resize({N, 1});
  dev_ctx.template Alloc<bool>(&bool_tensor_for_mask_label);

  // 准备一个和label同类型的tensor，每个元素都刷成ignore_index
  phi::DenseTensor ignore_label_as_tensor;

  const auto& label_type = labels->dtype();
  if (label_type == phi::DataType::INT32) {
    ignore_label_as_tensor.Resize({N, 1});
    dev_ctx.template Alloc<int>(&ignore_label_as_tensor);
    ret = xpu::constant<int>(dev_ctx.x_context(),
                             ignore_label_as_tensor.data<int>(),
                             N,
                             ignore_index);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
    // 如果label和ignore_index一样，那么把这个bool类型的对应位置刷成1，表示后面要刷成0
    // int equal(Context* ctx, const T* x, const T* y, bool* z, int64_t len);
    ret = xpu::equal<int>(dev_ctx.x_context(),
                          ignore_label_as_tensor.data<int>(),
                          labels->data<int>(),
                          bool_tensor_for_mask_label.data<bool>(),
                          N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "equal");
  } else if (label_type == phi::DataType::INT64) {
    ignore_label_as_tensor.Resize({N, 1});
    dev_ctx.template Alloc<int64_t>(&ignore_label_as_tensor);
    ret = xpu::constant<int64_t>(dev_ctx.x_context(),
                                 ignore_label_as_tensor.data<int64_t>(),
                                 N,
                                 ignore_index);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
    // 如果label和ignore_index一样，那么把这个bool类型的对应位置刷成1，表示后面要刷成0
    // int equal(Context* ctx, const T* x, const T* y, bool* z, int64_t len);
    ret = xpu::equal<int64_t>(dev_ctx.x_context(),
                              ignore_label_as_tensor.data<int64_t>(),
                              labels->data<int64_t>(),
                              bool_tensor_for_mask_label.data<bool>(),
                              N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "equal");
  }
  // bool值为1的说明命中了，要刷0，bool为0的要保留
  // int select(Context* ctx, const bool* condition, const T* x, const T* y,
  // T* z, const std::vector<int64_t>& condition_shape, const
  // std::vector<int64_t>& xshape);
  ret = xpu::select(
      dev_ctx.x_context(),
      reinterpret_cast<const bool*>(bool_tensor_for_mask_label.data<bool>()),
      reinterpret_cast<const XPUType*>(zeros_constant.data<T>()),
      reinterpret_cast<const XPUType*>(loss->data<T>()),
      reinterpret_cast<XPUType*>(loss->data<T>()),
      common::vectorize(predicted_logits->dims()),
      common::vectorize(predicted_logits->dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "select");
}
template <typename T>
struct CSoftmaxWithCrossEntropyFunctor<phi::XPUContext, T> {
  void operator()(const phi::XPUContext& dev_ctx,
                  const DenseTensor& logits_in,
                  const DenseTensor& label_in,
                  int64_t ignore_index,
                  int rank,
                  int nranks,
                  DenseTensor* softmax,
                  DenseTensor* loss) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    const phi::DenseTensor* logits = &logits_in;
    const phi::DenseTensor* labels = &label_in;

    XPUStream stream = nullptr;
    phi::distributed::BKCLCommContext* comm_ctx = nullptr;

    comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "BKCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    stream = dev_ctx.stream();

    // allocate memory on device.
    dev_ctx.template Alloc(softmax, logits->dtype());
    dev_ctx.template Alloc(loss, logits->dtype());

    const auto& logits_dims = logits->dims();

    const int axis = logits_dims.size() - 1;
    const int64_t N = phi::funcs::SizeToAxis(axis, logits_dims);
    const int64_t D = phi::funcs::SizeFromAxis(axis, logits_dims);

    phi::DenseTensor logits_2d, softmax_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});

    int ret = -1;
    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max.Resize({N, 1});
    dev_ctx.template Alloc<T>(&logits_max);
    {
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const T* x,
                  T* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_max<XPUType>(ctx,
                                        reinterpret_cast<const XPUType*>(x),
                                        reinterpret_cast<XPUType*>(y),
                                        xdims,
                                        reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, T>(
          dev_ctx,
          logits_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &logits_max,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_max");
    }
    comm_ctx->AllReduce(&logits_max, logits_max, BKCL_MAX, stream);

    // step 2, obtain logit - logit_max
    {
      auto f = [](xpu::Context* ctx,
                  const XPUType* x,
                  const XPUType* y,
                  XPUType* z,
                  const std::vector<int>& xshape,
                  const std::vector<int>& yshape) {
        return xpu::broadcast_sub<XPUType>(ctx, x, y, z, xshape, yshape);
      };
      phi::XPUElementwise<T, XPUType>(
          dev_ctx, logits_2d, logits_max, axis, &softmax_2d, f);
    }

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits.Resize({N, 1});
    dev_ctx.template Alloc<T>(&predicted_logits);
    ret = xpu::constant<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
        N,
        0.0);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
    const int64_t start_index = rank * D;
    const int64_t end_index = start_index + D;
    const auto& label_type = labels->dtype();
    if (label_type == phi::DataType::INT32) {
      ret = xpu::mask_label_by_index<XPUType, int32_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int32_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks,
          ignore_index);
    } else if (label_type == phi::DataType::INT64) {
      ret = xpu::mask_label_by_index<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          labels->data<int64_t>(),
          reinterpret_cast<XPUType*>(predicted_logits.data<T>()),
          start_index,
          end_index,
          N,
          D,
          nranks,
          ignore_index);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "mask_label_by_index");

    comm_ctx->AllReduce(&predicted_logits, predicted_logits, BKCL_ADD, stream);

    // step 4, obtain exp(logit)
    ret = xpu::exp<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
        reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
        N * D);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "exp");

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits.Resize({N, 1});
    dev_ctx.Alloc<T>(&sum_exp_logits);
    {
      int dims[1] = {1};
      auto f = [](xpu::Context* ctx,
                  const T* x,
                  T* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
        return xpu::reduce_sum<XPUType>(ctx,
                                        reinterpret_cast<const XPUType*>(x),
                                        reinterpret_cast<XPUType*>(y),
                                        xdims,
                                        reduce_dims);
      };
      ret = phi::XPUReduce<phi::XPUContext, T>(
          dev_ctx,
          softmax_2d,
          std::vector<int64_t>(dims, dims + 1),
          false,
          false,
          &sum_exp_logits,
          f);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
    }

    comm_ctx->AllReduce(&sum_exp_logits, sum_exp_logits, BKCL_ADD, stream);

    {
      int64_t dims[4] = {N, D, N, 1};
      ret = xpu::broadcast_div<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(softmax_2d.data<T>()),
          reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
          reinterpret_cast<XPUType*>(softmax_2d.data<T>()),
          std::vector<int64_t>(dims, dims + 2),
          std::vector<int64_t>(dims + 2, dims + 4));
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast_div");
    }

    ret = xpu::log<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<XPUType*>(sum_exp_logits.data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "log");
    ret = xpu::sub<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(sum_exp_logits.data<T>()),
        reinterpret_cast<const XPUType*>(predicted_logits.data<T>()),
        reinterpret_cast<XPUType*>(loss->data<T>()),
        N * 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sub");

    // 将label和ignore_index相同的那些loss，置为0
    FixLossAccordingToIgnoreIndex<T>(
        dev_ctx, labels, &predicted_logits, loss, N, ignore_index);
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_cross_entropy,
                   XPU,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithCrossEntropyKernel,
                   float,
                   phi::dtype::bfloat16) {}
