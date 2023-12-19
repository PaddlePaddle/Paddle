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

#include "paddle/phi/kernels/hsigmoid_loss_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "paddle/phi/kernels/funcs/matrix_bit_code.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void HSigmoidLossKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& label,
                        const DenseTensor& w,
                        const paddle::optional<DenseTensor>& bias,
                        const paddle::optional<DenseTensor>& path,
                        const paddle::optional<DenseTensor>& code,
                        int num_classes,
                        bool is_sparse,
                        DenseTensor* out,
                        DenseTensor* pre_out,
                        DenseTensor* w_out) {
  size_t num_classes_st = static_cast<size_t>(num_classes);
  // for remote prefetch

  bool is_custom = false;
  if (path.get_ptr()) {
    is_custom = true;
  }
  int64_t code_length =
      path.get_ptr()
          ? static_cast<int64_t>(path.get_ptr()->dims()[1])
          : static_cast<int64_t>(phi::funcs::FindLastSet(num_classes_st - 1));
  int64_t batch_size = x.dims()[0];
  DenseTensor sum;
  pre_out->Resize(common::make_ddim({batch_size, code_length}));
  ctx.template Alloc<T>(pre_out);
  auto* pre_out_data = pre_out->data<T>();
  auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
  // Not all class(leaf) nodes' path lengths equal code_length, thus init as
  // 0s can avoid out of path's loss.
  funcs::SetConstant<Context, T> zero;
  zero(ctx, pre_out, static_cast<T>(0.0));
  auto& place = *ctx.eigen_device();
  funcs::RowwiseSum<Context, T> row_sum;

  std::unique_ptr<phi::funcs::MatrixBitCodeFunctor<T>> bit_code;
  if (!is_custom) {
    bit_code.reset(new phi::funcs::MatrixBitCodeFunctor<T>(
        num_classes_st, label.template data<int64_t>()));
  } else {
    bit_code.reset(new phi::funcs::MatrixBitCodeFunctor<T>(
        *(path.get_ptr()), *(code.get_ptr()), label.template data<int64_t>()));
  }

  std::vector<int64_t> sum_dims({batch_size, 1UL});
  sum.Resize(common::make_ddim(sum_dims));
  ctx.template Alloc<T>(&sum);
  auto sum_mat = EigenMatrix<T>::From(sum);
  ctx.template Alloc<T>(out);
  auto out_mat = EigenMatrix<T>::From(*out);
  if (bias.get_ptr()) {
    bit_code->Add(*(bias.get_ptr()), pre_out);
  }
  bit_code->Mul(pre_out, w, x);
  // clip to [-40, 40]
  phi::Transform<Context> trans;
  trans(ctx,
        pre_out_data,
        pre_out_data + pre_out->numel(),
        pre_out_data,
        ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
  bit_code->Sum(*pre_out, out, static_cast<T>(-1));
  // use softrelu to calculate cross entropy
  pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();
  row_sum(ctx, *pre_out, &sum);
  // TODO(guosheng): Subtract the out of path's loss, since not all
  // class(leaf) nodes' path lengths equal code_length. But it won't break the
  // gradient check since both have the out of path's loss and will cancel out
  // each other.
  out_mat.device(place) = sum_mat + out_mat;
}

}  // namespace phi

PD_REGISTER_KERNEL(
    hsigmoid_loss, CPU, ALL_LAYOUT, phi::HSigmoidLossKernel, float, double) {}
