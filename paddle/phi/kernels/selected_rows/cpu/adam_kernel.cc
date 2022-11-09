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
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {
namespace sr {

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
    const paddle::optional<DenseTensor>& master_param,
    const paddle::optional<DenseTensor>& skip_update,
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

  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  T epsilon_ = epsilon.to<T>();

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel();
  VLOG(3) << "beta2_pow.numel() : " << beta2_pow.numel();
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
    phi::funcs::scatter::MergeAdd<Context, T> merge_func;
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

  funcs::SparseAdamFunctor<T, funcs::CPUAdam> functor(
      beta1_,
      beta2_,
      epsilon_,
      beta1_pow.data<T>(),
      beta2_pow.data<T>(),
      moment1.data<T>(),
      dev_ctx.template Alloc<T>(moment1_out),
      moment2.data<T>(),
      dev_ctx.template Alloc<T>(moment2_out),
      learning_rate.data<T>(),
      grad_data,
      param.data<T>(),
      dev_ctx.template Alloc<T>(param_out),
      rows,
      row_numel,
      grad_merge.rows().size(),
      lazy_mode);
  // update beta1 and beta2
  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out)[0] =
        beta1_ * beta1_pow.data<T>()[0];
    dev_ctx.template Alloc<T>(beta2_pow_out)[0] =
        beta2_ * beta2_pow.data<T>()[0];
  }
  if (lazy_mode) {
    VLOG(3) << "run cpu lazy mode";
    size_t row_count = grad_merge.rows().size();
    std::vector<int64_t> cpu_rows(grad_merge.rows());
    for (size_t row_index = 0; row_index < row_count; ++row_index) {
      for (size_t offset = 0; offset < row_numel; ++offset) {
        size_t i = cpu_rows[row_index] * row_numel + offset;
        functor.adam_update(i, grad_data[row_index * row_numel + offset]);
      }
    }
  }
#ifndef _WIN32
  else if (FLAGS_inner_op_parallelism > 1 &&  // NOLINT
           min_row_size_to_use_multithread > 0 &&
           param.dims()[0] > min_row_size_to_use_multithread) {
    VLOG(3) << "use multi thread, inner_op_parallelism="
            << FLAGS_inner_op_parallelism << " min_row_size_to_use_multithread="
            << min_row_size_to_use_multithread;
    if (FLAGS_inner_op_parallelism > 10) {
      VLOG(1) << "FLAGS_inner_op_parallelism " << FLAGS_inner_op_parallelism
              << " is two large!";
    }
    auto& grad_rows = grad_merge.rows();
    std::unordered_map<size_t, int> row_id_to_grad_row_offset;
    size_t param_row_count = param.numel() / row_numel;
    if (param_row_count < 1000) {
      VLOG(1) << "param_row_count should be larger then 1000 to use "
                 "multi thread, currently "
              << param_row_count;
    }
    for (size_t i = 0; i < grad_rows.size(); ++i) {
      row_id_to_grad_row_offset[grad_rows[i]] = i;
    }
    std::vector<std::future<void>> fs;
    int64_t line_in_each_thread =
        param_row_count / FLAGS_inner_op_parallelism + 1;
    for (int i = 0; i < FLAGS_inner_op_parallelism; ++i) {
      int64_t start = i * line_in_each_thread;
      int64_t end = (i + 1) * line_in_each_thread;
      if (start >= static_cast<int64_t>(param_row_count)) {
        break;
      }
      if (end > static_cast<int64_t>(param_row_count)) {
        end = static_cast<int64_t>(param_row_count);
      }
      fs.push_back(paddle::framework::Async([&functor,
                                             &row_id_to_grad_row_offset,
                                             &grad_data,
                                             row_numel,
                                             start,
                                             end]() {
        for (int64_t row_id = start; row_id < end; ++row_id) {
          auto iter = row_id_to_grad_row_offset.find(row_id);
          if (iter != row_id_to_grad_row_offset.end()) {
            for (size_t row_offset = 0U; row_offset < row_numel; ++row_offset) {
              functor.adam_update(
                  row_id * row_numel + row_offset,
                  grad_data[iter->second * row_numel + row_offset]);
            }
          } else {
            for (size_t row_offset = 0U; row_offset < row_numel; ++row_offset) {
              functor.adam_update(row_id * row_numel + row_offset, 0);
            }
          }
        }
      }));
    }
    for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
  }
#endif    // !_WIN32
  else {  // NOLINT
    functor(param.numel());
  }
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(adam_dense_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::AdamDenseParamSparseGradKernel,
                   float,
                   double) {}
