// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace phi {

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";

template <typename T, typename Context>
void ShadowFeedKernel(const Context& ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  if (!x.initialized()) {
    ctx.template Alloc<T>(out);
    return;
  }
  if (x.place() == ctx.GetPlace()) {
    out->ShareDataWith(x);
    out->set_lod(x.lod());
  } else {
    phi::Copy<Context>(ctx, x, ctx.GetPlace(), true, out);
  }
}

template <typename T, typename Context>
void ShadowFeedTensorsKernel(const Context& ctx,
                             const std::vector<const DenseTensor*>& xs,
                             std::vector<DenseTensor*> outs) {
  for (size_t i = 0; i < xs.size(); ++i) {
    ShadowFeedKernel<T, Context>(ctx, *(xs[i]), outs[i]);
  }
}

template <typename T, typename Context>
void PrintKernel(const Context& ctx,
                 const DenseTensor& x,
                 int first_n,
                 const std::string& message,
                 int summarize,
                 bool print_tensor_name,
                 bool print_tensor_type,
                 bool print_tensor_shape,
                 bool print_tensor_layout,
                 bool print_tensor_lod,
                 const std::string& print_phase,
                 bool is_forward,
                 DenseTensor* out) {
  phi::Copy<Context>(ctx, x, ctx.GetPlace(), true, out);
  out->set_lod(x.lod());

  if ((is_forward && print_phase == kBackward) ||
      (!is_forward && print_phase == kForward)) {
    return;
  }

  // TODO(phlrain): support first_n using a input tensor
  // if (first_n > 0 && ++times_ > first_n) return;

  // TODO(phlrain): support printed_var_name
  paddle::funcs::TensorFormatter formatter;
  const std::string& name = print_tensor_name ? "var" : "";
  formatter.SetPrintTensorType(print_tensor_type);
  formatter.SetPrintTensorShape(print_tensor_shape);
  formatter.SetPrintTensorLod(print_tensor_lod);
  formatter.SetPrintTensorLayout(print_tensor_layout);
  formatter.SetSummarize(summarize);
  formatter.Print(x, name, message);
}

}  // namespace phi
