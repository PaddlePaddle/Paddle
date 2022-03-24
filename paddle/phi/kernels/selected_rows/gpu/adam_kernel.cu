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

#include "paddle/phi/kernels/selected_rows/adam_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
namespace sr {

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T, typename MT>
__global__ void SparseAdamCUDAKernelREG(MT beta1,
                                        MT beta2,
                                        MT epsilon,
                                        const MT beta1_pow,
                                        const MT beta2_pow,
                                        const MT* mom1_,
                                        MT* mom1_out_,
                                        const MT* mom2_,
                                        MT* mom2_out_,
                                        const MT* lr_,
                                        const T* grad_,
                                        const T* param_,
                                        T* param_out_,
                                        const MT* master_param,
                                        MT* master_param_out,
                                        const int64_t* rows_,
                                        int64_t row_numel,
                                        int64_t row_count,
                                        bool lazy_mode,
                                        int ndim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  MT lr = *lr_;

  for (; id < ndim; id += blockDim.x * gridDim.x) {
    auto row_idx =
        phi::funcs::BinarySearch<int64_t>(rows_, row_count, id / row_numel);
    if (lazy_mode && row_idx < 0) {
      return;
    } else {
      MT mom1 = mom1_[id];
      MT mom2 = mom2_[id];
      MT p = master_param ? master_param[id] : static_cast<MT>(param_[id]);
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel + id % row_numel])
                 : static_cast<MT>(0);
      mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
      mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

      MT denom =
          (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
      p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

      // Write back to global memory
      mom1_out_[id] = mom1;
      mom2_out_[id] = mom2;
      param_out_[id] = static_cast<T>(p);
      if (master_param_out) {
        master_param_out[id] = p;
      }
    }
  }
}

template <typename T, typename Context>
void AdamDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const SelectedRows& grad,
    const DenseTensor& learning_rate,
    const DenseTensor& moment1,
    const DenseTensor& moment2,
    const DenseTensor& beta1_pow,
    const DenseTensor& beta2_pow,
    paddle::optional<const DenseTensor&> master_param,
    paddle::optional<const DenseTensor&> skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    bool use_global_beta_pow,
    DenseTensor* param_out,
    DenseTensor* moment1_out,
    DenseTensor* moment2_out,
    DenseTensor* beta1_pow_out,
    DenseTensor* beta2_pow_out,
    DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    paddle::framework::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }
  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    phi::Copy(dev_ctx, beta1_pow, dev_ctx.GetPlace(), false, beta1_pow_out);
    phi::Copy(dev_ctx, beta2_pow, dev_ctx.GetPlace(), false, beta2_pow_out);
    return;
  }

  MPDType beta1_ = beta1.to<MPDType>();
  MPDType beta2_ = beta2.to<MPDType>();
  MPDType epsilon_ = epsilon.to<MPDType>();
  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
          << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();
  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  const MPDType* master_in_data =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  if (grad.rows().size() == 0) {
    VLOG(3) << "grad row size is 0!!";
    return;
  }

  std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
  bool is_strict_sorted = true;
  for (size_t i = 1; i < cpu_rows.size(); ++i) {
    if (cpu_rows[i - 1] >= cpu_rows[i]) {
      is_strict_sorted = false;
      break;
    }
  }

  phi::SelectedRows tmp_grad_merge;
  const phi::SelectedRows* grad_merge_ptr;
  if (is_strict_sorted) {
    grad_merge_ptr = &grad;
  } else {
    // merge duplicated rows if any.
    // The rows of grad_merge have been sorted inside MergeAdd functor
    paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
    merge_func(dev_ctx, grad, &tmp_grad_merge, true);
    grad_merge_ptr = &tmp_grad_merge;
  }
  auto& grad_merge = *grad_merge_ptr;
  auto& grad_tensor = grad_merge.value();
  const T* grad_data = grad_tensor.template data<T>();
  auto* grad_merge_rows = &grad_merge.rows();
  paddle::framework::MixVector<int64_t> mixv_grad_merge_rows(grad_merge_rows);
  const int64_t* rows = mixv_grad_merge_rows.Data(dev_ctx.GetPlace());
  auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    int threads = 512;
    int ndim = param.numel();
    int blocks = (ndim + threads - 1) / threads;

    SparseAdamCUDAKernelREG<T,
                            MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
        beta1_,
        beta2_,
        epsilon_,
        *beta1_pow.data<MPDType>(),
        *beta2_pow.data<MPDType>(),
        moment1.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment1_out),
        moment2.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment2_out),
        learning_rate.data<MPDType>(),
        grad_data,
        param.data<T>(),
        dev_ctx.template Alloc<T>(param_out),
        master_in_data,
        master_out_data,
        rows,
        row_numel,
        grad_merge.rows().size(),
        lazy_mode,
        ndim);
    if (!use_global_beta_pow) {
      // Update with cpu
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1_ * beta1_pow.data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2_ * beta2_pow.data<MPDType>()[0];
    }
  } else {
    funcs::SparseAdamFunctor<T, funcs::GPUAdam, MPDType> functor(
        beta1_,
        beta2_,
        epsilon_,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        moment1.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment1_out),
        moment2.data<MPDType>(),
        dev_ctx.template Alloc<MPDType>(moment2_out),
        learning_rate.data<MPDType>(),
        grad_data,
        param.data<T>(),
        dev_ctx.template Alloc<T>(param_out),
        master_in_data,
        master_out_data,
        rows,
        row_numel,
        grad_merge.rows().size(),
        lazy_mode);

    // FIXME(minqiyang): remove BinarySearch in GPU later
    funcs::ForRange<Context> for_range(dev_ctx, param.numel());
    for_range(functor);
    if (!use_global_beta_pow) {
      // update beta1 and beta2
      UpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          beta1_pow.data<MPDType>(),
          beta2_pow.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(beta1_pow_out),
          dev_ctx.template Alloc<MPDType>(beta2_pow_out));
    }
  }
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(adam_dense_param_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::AdamDenseParamSparseGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
