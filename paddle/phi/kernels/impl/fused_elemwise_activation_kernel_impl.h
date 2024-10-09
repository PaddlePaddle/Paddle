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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/compound_functors.h"
#include "paddle/phi/kernels/funcs/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/funcs/fused_elemwise_activation_functor.h"

namespace phi {

template <typename T, typename Context>
void FusedElemwiseActivationKernel(const Context &dev_ctx,
                                   const DenseTensor &x,
                                   const DenseTensor &y,
                                   const std::vector<std::string> &functor_list,
                                   int axis,
                                   float scale,
                                   bool save_intermediate_out,
                                   DenseTensor *out,
                                   DenseTensor *intermediate_out) {
  auto &in_x = GET_DATA_SAFELY(&x, "Input", "X", "FusedElemwiseActivation");
  auto &in_y = GET_DATA_SAFELY(&y, "Input", "Y", "FusedElemwiseActivation");

  PADDLE_ENFORCE_EQ(
      out != nullptr,
      true,
      common::errors::InvalidArgument("The output(Out) should not be empty"));
  auto output = out;

  std::vector<phi::DenseTensor *> outputs;
  outputs.emplace_back(output);

  if (save_intermediate_out) {
    PADDLE_ENFORCE_EQ(intermediate_out != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "The save_intermediate_out is enable, so the "
                          "IntermediateOut should not be empty."));

    outputs.emplace_back(intermediate_out);
  } else {
    outputs.emplace_back(nullptr);
  }

  phi::funcs::RunFunctors<Context, T>(dev_ctx,
                                      in_x,
                                      in_y,
                                      &outputs,
                                      functor_list,
                                      scale,
                                      axis,
                                      save_intermediate_out);
}

template <typename T, typename Context>
void FusedElemwiseActivationGradKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &y,
    const DenseTensor &out,
    const DenseTensor &intermediate_out,
    const DenseTensor &out_grad,
    const std::vector<std::string> &functor_list,
    int axis,
    float scale,
    bool save_intermediate_out,
    DenseTensor *x_grad,
    DenseTensor *y_grad) {
  auto *in_y = &y;
  PADDLE_ENFORCE_NE(
      in_y,
      nullptr,
      common::errors::InvalidArgument("Input(Y) should not be nullptr."));
  phi::DenseTensor *in_out = const_cast<phi::DenseTensor *>(&out);

  auto in_out_grad = &out_grad;
  PADDLE_ENFORCE_NE(in_out_grad,
                    nullptr,
                    common::errors::InvalidArgument(
                        "Input(Out@Grad) should not be nullptr."));

  std::vector<std::string> functor_list_new = functor_list;
  size_t sz = functor_list_new[0].size();
  int start = sz < 5 ? 0 : (sz - 5);
  if (functor_list_new[0].substr(start, 5) != "_grad") {
    functor_list_new[0] += "_grad";
  }
  sz = functor_list_new[1].size();
  start = sz < 5 ? 0 : (sz - 5);
  if (functor_list_new[1].substr(start, 5) != "_grad") {
    functor_list_new[1] += "_grad";
  }

  phi::DenseTensor *in_x = const_cast<phi::DenseTensor *>(&x);
  phi::DenseTensor *d_intermediate_out =
      nullptr;  // intermediate_out_grad  is not supported in ops.yaml, so use
                // nullptr

  // Get intermediate_out
  phi::DenseTensor *in_intermediate_out = nullptr;
  if (save_intermediate_out) {
    // if save_intermediate_out is true, for Unary(Binary(x, y)) and
    // Binary(x, Unary(y)), the Binary(x, y) and Unary(y) not need to
    // recompute.
    in_intermediate_out = const_cast<phi::DenseTensor *>(&intermediate_out);
    PADDLE_ENFORCE_NE(in_intermediate_out,
                      nullptr,
                      common::errors::InvalidArgument(
                          "The option of 'save_intermediate_out' is opened,"
                          " so the number of 'Out' should be two."));
  } else {
    if (!phi::funcs::InputXCanBeAbsent(functor_list_new)) {
      PADDLE_ENFORCE_NE(
          in_x,
          nullptr,
          common::errors::InvalidArgument("Input(X) should not be null."));
    }
  }

  // Get in_x
  if (x.initialized()) {
    PADDLE_ENFORCE_NE(
        in_x,
        nullptr,
        common::errors::InvalidArgument("Input(X) should not be null."));
  } else {
    // If functor_list contains elementwise_add, the backward doesn't use
    // in_x, in_y and in_out.
    PADDLE_ENFORCE_EQ(phi::funcs::InputXCanBeAbsent(functor_list_new),
                      true,
                      common::errors::InvalidArgument(
                          "Only when the compoundfunctor contains "
                          "elementwise_add_grad, the 'X' could be absent."));
    in_x = const_cast<phi::DenseTensor *>(in_out_grad);
  }

  // Get in_Out
  if (out.initialized()) {
    PADDLE_ENFORCE_NE(
        in_out,
        nullptr,
        common::errors::InvalidArgument("Input(X) should not be null."));
  } else {
    // If functor_list contains elementwise_add, the backward doesn't use
    // in_x, in_y and in_out.
    PADDLE_ENFORCE_EQ(phi::funcs::InputXCanBeAbsent(functor_list_new),
                      true,
                      common::errors::InvalidArgument(
                          "Only when the compoundfunctor contains "
                          "elementwise_add_grad, the 'X' could be absent."));
    in_out = const_cast<phi::DenseTensor *>(in_out_grad);
  }

  bool has_in_place = phi::funcs::HasInPlaceUnary(functor_list_new);
  if (has_in_place) {
    phi::funcs::RunGradFunctors<Context, T, true /*InPlace*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        x_grad,
        y_grad,
        d_intermediate_out,
        functor_list_new,
        scale,
        axis);
  } else {
    phi::funcs::RunGradFunctors<Context, T, false /*InPlace*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        x_grad,
        y_grad,
        d_intermediate_out,
        functor_list_new,
        scale,
        axis);
  }
}

template <typename T, typename Context>
void FusedElemwiseAddActivationKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &y,
    const std::vector<std::string> &functor_list,
    int axis,
    float scale,
    bool save_intermediate_out,
    DenseTensor *out,
    DenseTensor *intermediate_out) {
  FusedElemwiseActivationKernel<T, Context>(dev_ctx,
                                            x,
                                            y,
                                            functor_list,
                                            axis,
                                            scale,
                                            save_intermediate_out,
                                            out,
                                            intermediate_out);
}

template <typename T, typename Context>
void FusedElemwiseAddActivationGradKernel(
    const Context &dev_ctx,
    const paddle::optional<DenseTensor> &x,
    const DenseTensor &y,
    const DenseTensor &out,
    const paddle::optional<DenseTensor> &intermediate_out,
    const DenseTensor &out_grad,
    const std::vector<std::string> &functor_list,
    int axis,
    float scale,
    bool save_intermediate_out,
    DenseTensor *x_grad,
    DenseTensor *y_grad) {
  phi::DenseTensor tmp_x;
  phi::DenseTensor tmp_i;
  if (x) {
    tmp_x = x.get();
  }
  if (intermediate_out) {
    tmp_i = intermediate_out.get();
  }
  FusedElemwiseActivationGradKernel<T, Context>(dev_ctx,
                                                tmp_x,
                                                y,
                                                out,
                                                tmp_i,
                                                out_grad,
                                                functor_list,
                                                axis,
                                                scale,
                                                save_intermediate_out,
                                                x_grad,
                                                y_grad);
}

}  // namespace phi
