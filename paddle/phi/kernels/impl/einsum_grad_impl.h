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
#pragma once

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/impl/einsum_impl.h"
#include "paddle/phi/kernels/tile_grad_kernel.h"
#include "paddle/phi/kernels/tile_kernel.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {

template <typename T, typename Context>
DenseTensor PerformTileAndReduction(const Context& dev_ctx,
                                    const LabelMap& label2type,
                                    const LabelMap& label2shape,
                                    const std::vector<int>& broadcast_shape,
                                    const std::vector<int> x_shape,
                                    std::string equ,   // value pass
                                    DenseTensor& t) {  // NOLINT
  auto tmp_label = equ;
  auto tmp_union = unique_labels(tmp_label);
  auto op_label = std::string(tmp_union.begin(), tmp_union.end());
  VLOG(5) << "Start PerformTileAndReduction equation " << equ
          << " with operand shape: "
          << paddle::string::join_strings(common::vectorize<int>(t.dims()),
                                          ",");
  DenseTensor ret;
  std::vector<int> repeat_times;
  std::vector<int> resize_dims;
  std::vector<int> recover_shape;
  for (int c : op_label) {
    if (label2type[c] == LabelType::Reduction) {
      repeat_times.push_back(label2shape[c]);
      resize_dims.push_back(1);
      recover_shape.push_back(label2shape[c]);
    } else {
      resize_dims.push_back(label2shape[c]);
      repeat_times.push_back(1);
      recover_shape.push_back(label2shape[c]);
    }
  }
  t.Resize(common::make_ddim(resize_dims));
  DenseTensor after_tile;
  if (std::all_of(repeat_times.begin(), repeat_times.end(), [](int x) {
        return x == 1;
      })) {
    after_tile = t;
  } else {
    VLOG(4) << "do TileKernel with repeat_times="
            << paddle::string::join_strings(repeat_times, ",");
    TileKernel<T, Context>(dev_ctx, t, repeat_times, &after_tile);
  }
  ret = after_tile;
  VLOG(5) << "PermformTileAndReduction: recover shape: "
          << paddle::string::join_strings(recover_shape, ",");
  ret.Resize(common::make_ddim(recover_shape));

  // undiagonalize by einsum equation. only contain undiagonal operations.
  DenseTensor undiagonal_out;
  if (op_label != equ) {
    VLOG(5) << "Undiagonal by einsum with args: " << op_label + "->" + equ;
    EinsumInferKernel<T, Context>(
        dev_ctx, {&ret}, op_label + "->" + equ, &undiagonal_out);
  } else {
    undiagonal_out = ret;
  }

  // call TileGradKernel to reverse broadcast operation.
  VLOG(5) << "After diagonalize, we have tensor with shape: "
          << paddle::string::join_strings(
                 common::vectorize<int>(undiagonal_out.dims()), ',');
  repeat_times.clear();
  for (size_t i = 0; i < x_shape.size(); ++i) {
    VLOG(4) << "broadcast shape is " << broadcast_shape[i] << ", x_shape is "
            << x_shape[i];
    repeat_times.push_back(broadcast_shape[i] / x_shape[i]);
  }
  bool is_all_ones = std::all_of(
      repeat_times.begin(), repeat_times.end(), [](int x) { return x == 1; });
  if (is_all_ones) {
    VLOG(4) << "don't need broadcast recover, we just return undiagonal_out.";
    return undiagonal_out;
  }
  DenseTensor tmp_x;
  DenseTensor broadcast_out;
  tmp_x.Resize(common::make_ddim(x_shape));
  broadcast_out.Resize(common::make_ddim(x_shape));
  TileGradKernel<T, Context>(
      dev_ctx, tmp_x, undiagonal_out, repeat_times, &broadcast_out);
  VLOG(5) << "After broadcast recover, we have tensor with shape: "
          << paddle::string::join_strings(
                 common::vectorize<int>(broadcast_out.dims()), ',');
  return broadcast_out;
}

template <typename T, typename Context>
void EinsumGradKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& x,
                      const std::vector<const DenseTensor*>& inner_cache,
                      const DenseTensor& out_grad,
                      const std::string& equation,
                      std::vector<DenseTensor*> x_grad) {
  VLOG(5) << "Start EinsumGradKernel:";
  LabelMap labelshape(0);
  LabelMap labeltype(LabelType::Reduction);
  std::vector<LabelMap> label2perms(x.size(), LabelMap(-1));
  std::vector<char> all_labels;  // order: ABO, AO, BO, AB, Reduce
  std::vector<std::vector<int>> broadcast_shapes(2);
  std::vector<int> output_dims;

  std::vector<DDim> input_dims;
  for (auto& i : x) {
    input_dims.push_back(i->dims());
  }
  std::vector<std::string> input_strs;
  std::string right;
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &broadcast_shapes,
                      &output_dims,
                      &right,
                      &input_strs);

  VLOG(4) << "After grad parse einsum equation.";

  auto gather_labels_except_reduction = [&labeltype](std::string all) {
    std::string res("");
    for (auto c : all)
      if (labeltype[static_cast<int>(c)] != LabelType::Reduction) res += c;
    auto tmp_unique = unique_labels(res);
    return std::string(tmp_unique.begin(), tmp_unique.end());
  };
  if (x.size() == 1) {  // Unary
    auto splits = paddle::string::split_string(equation, "->");
    auto left = splits[0];
    right = splits[1];
    auto new_equation = right + "->" + gather_labels_except_reduction(left);
    auto new_operands = std::vector<const DenseTensor*>();
    new_operands.push_back(&out_grad);
    DenseTensor before_tile;
    VLOG(5) << "new_equation is " << new_equation;
    EinsumInferKernel<T, Context>(
        dev_ctx, new_operands, new_equation, &before_tile);
    *(x_grad[0]) = PerformTileAndReduction<T, Context>(
        dev_ctx,
        labeltype,
        labelshape,
        broadcast_shapes[0],
        common::vectorize<int>(x[0]->dims()),
        left,
        before_tile);
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
    *(x_grad[0]) = Conj<T, Context>(dev_ctx, *x_grad[0]);
#endif
  } else {
    auto splits = paddle::string::split_string(equation, "->");
    auto left = splits[0];
    auto ops = paddle::string::split_string(left, ",");
    right = splits[1];
    auto equation_for_A =
        ops[1] + "," + right + "->" + gather_labels_except_reduction(ops[0]);
    auto equation_for_B =
        right + "," + ops[0] + "->" + gather_labels_except_reduction(ops[1]);
    auto operands_for_A = std::vector<const DenseTensor*>();
    auto operands_for_B = std::vector<const DenseTensor*>();
    DenseTensor dA, dB;
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
    auto out_grad_conj = Conj<T, Context>(dev_ctx, out_grad);
#else
    auto out_grad_conj = out_grad;
#endif
    // dA = einsum(B, dC)
    operands_for_A.push_back(x[1]);
    operands_for_A.push_back(&out_grad_conj);
    // dB = einsum(dC, A)
    operands_for_B.push_back(&out_grad_conj);
    operands_for_B.push_back(x[0]);

    std::vector<DenseTensor> cache(3);  // set empty; TA, TB, TdC
    if (inner_cache.size() >
        0) {  // for compatibility,  we can load and run v2.3 EinsumOp.
      cache[0].ShareBufferWith(*(inner_cache[0]));
      cache[1].ShareBufferWith(*(inner_cache[1]));
    }
    EinsumKernelImpl<T, Context>(dev_ctx,
                                 all_labels,
                                 operands_for_A,
                                 equation_for_A,
                                 &dA,
                                 {&cache[1], &cache[2]},
                                 false);

    EinsumKernelImpl<T, Context>(dev_ctx,
                                 all_labels,
                                 operands_for_B,
                                 equation_for_B,
                                 &dB,
                                 {&cache[2], &cache[0]},
                                 false);

    // release the cache tensor dTC to save memory right now. they are useless
    // now.
    cache.clear();
    if (x_grad[0]) {
      *(x_grad[0]) = PerformTileAndReduction<T, Context>(
          dev_ctx,
          labeltype,
          labelshape,
          broadcast_shapes[0],
          common::vectorize<int>(x[0]->dims()),
          ops[0],
          dA);
      VLOG(4) << "After call dA";
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
      *(x_grad[0]) = Conj<T, Context>(dev_ctx, *x_grad[0]);
#endif
    }
    if (x_grad[1]) {
      *(x_grad[1]) = PerformTileAndReduction<T, Context>(
          dev_ctx,
          labeltype,
          labelshape,
          broadcast_shapes[1],
          common::vectorize<int>(x[1]->dims()),
          ops[1],
          dB);
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
      *(x_grad[1]) = Conj<T, Context>(dev_ctx, *x_grad[1]);
#endif
      VLOG(4) << "After call dA";
    }
  }
}
}  // namespace phi
