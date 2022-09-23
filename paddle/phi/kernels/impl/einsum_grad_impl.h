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

#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/impl/einsum_impl.h"
#include "paddle/phi/kernels/tile_kernel.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {
template <typename T, typename Context>
DenseTensor PerformTileAndReduction(const Context& dev_ctx,
                                    const LabelMap& label2type,
                                    const LabelMap& label2shape,
                                    const std::vector<int>& broadcast_dims,
                                    const std::vector<int>& ellipsis_dims,
                                    std::string op_label,  // value pass
                                    DenseTensor& t) {      // NOLINT
  ReplaceEllipsis(op_label);
  DenseTensor ret;
  std::vector<int> repeat_times;
  std::vector<int> resize_dims;
  std::vector<int> recover_shape;
  for (int c : op_label) {
    if (label2type[c] == LabelType::Reduction) {
      // '.' can't be Reduction, so we don't deal '.' here.
      repeat_times.push_back(label2shape[c]);
      resize_dims.push_back(1);
      recover_shape.push_back(label2shape[c]);
    } else {
      if (c != '.') {
        resize_dims.push_back(label2shape[c]);
        repeat_times.push_back(1);
        recover_shape.push_back(label2shape[c]);
      } else {
        int n_dims = broadcast_dims.size();
        resize_dims.insert(
            resize_dims.end(), broadcast_dims.begin(), broadcast_dims.end());
        recover_shape.insert(
            recover_shape.end(), ellipsis_dims.begin(), ellipsis_dims.end());
        while (n_dims--) repeat_times.push_back(1);
      }
    }
  }
  t.Resize(make_ddim(resize_dims));
  DenseTensor after_tile;
  if (std::all_of(repeat_times.begin(), repeat_times.end(), [](int x) {
        return x == 1;
      })) {
    after_tile = t;
  } else {
    TileKernel<T, Context>(dev_ctx, t, repeat_times, &after_tile);
  }
  size_t n_ellipsis_idx = op_label.find(".", 0);
  if (n_ellipsis_idx != std::string::npos) {
    // may be we need reduce. broadcast_dims is not equal to ellipsis dims.
    std::vector<int64_t> to_reduce;
    for (size_t i = 0; i < broadcast_dims.size() - ellipsis_dims.size(); ++i)
      to_reduce.push_back(i + n_ellipsis_idx);

    int new_offset =
        n_ellipsis_idx + broadcast_dims.size() - ellipsis_dims.size();
    for (size_t i = 0; i < ellipsis_dims.size(); ++i)
      if (ellipsis_dims[i] == 1) to_reduce.push_back(i + new_offset);

    VLOG(5) << "PermformTileAndReduction: reduce sum axis: "
            << paddle::string::join_strings(to_reduce, ",");
    if (to_reduce.size() != 0) {
      ret = Sum<T, Context>(dev_ctx,
                            after_tile,
                            phi::IntArray(to_reduce),
                            after_tile.dtype(),
                            false);  // not keep dim.
    } else {
      ret = after_tile;
    }
  } else {
    ret = after_tile;
  }
  VLOG(5) << "PermformTileAndReduction: recover shape: "
          << paddle::string::join_strings(recover_shape, ",");
  ret.Resize(make_ddim(recover_shape));
  return ret;
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
  std::vector<std::vector<int>> ellipsis_dims(2);
  std::vector<int> broadcast_dims;
  std::vector<int> output_dims;

  std::vector<DDim> input_dims;
  for (auto& i : x) {
    input_dims.push_back(i->dims());
  }
  std::string right;
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &ellipsis_dims,
                      &broadcast_dims,
                      &output_dims,
                      &right);

  auto gather_labels_except_reduction = [&labeltype](std::string all) {
    std::string res("");
    for (auto c : all)
      if (labeltype[static_cast<int>(c)] != LabelType::Reduction) res += c;
    return res;
  };
  if (x.size() == 1) {  // Unary
    auto splits = paddle::string::split_string(equation, "->");
    auto left = splits[0];
    right = splits[1].substr(1);
    auto new_equation = right + "->" + gather_labels_except_reduction(left);
    auto new_operands = std::vector<const DenseTensor*>();
    new_operands.push_back(&out_grad);
    DenseTensor before_tile;
    EinsumKernel<T, Context>(dev_ctx, new_operands, new_equation, &before_tile);
    *(x_grad[0]) = PerformTileAndReduction<T, Context>(dev_ctx,
                                                       labeltype,
                                                       labelshape,
                                                       broadcast_dims,
                                                       ellipsis_dims[0],
                                                       left,
                                                       before_tile);
  } else {
    auto splits = paddle::string::split_string(equation, "->");
    auto left = splits[0];
    auto ops = paddle::string::split_string(left, ",");
    right = splits[1].substr(1);

    auto equation_for_A =
        ops[1] + "," + right + "->" + gather_labels_except_reduction(ops[0]);
    auto equation_for_B =
        right + "," + ops[0] + "->" + gather_labels_except_reduction(ops[1]);
    auto operands_for_A = std::vector<const DenseTensor*>();
    auto operands_for_B = std::vector<const DenseTensor*>();
    DenseTensor dA, dB;
    // dA = einsum(B, dC)
    operands_for_A.push_back(x[1]);
    operands_for_A.push_back(&out_grad);
    // dB = einsum(dC, A)
    operands_for_B.push_back(&out_grad);
    operands_for_B.push_back(x[0]);

    DenseTensor before_tile;

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
      *(x_grad[0]) = PerformTileAndReduction<T, Context>(dev_ctx,
                                                         labeltype,
                                                         labelshape,
                                                         broadcast_dims,
                                                         ellipsis_dims[0],
                                                         ops[0],
                                                         dA);
    }
    if (x_grad[1]) {
      *(x_grad[1]) = PerformTileAndReduction<T, Context>(dev_ctx,
                                                         labeltype,
                                                         labelshape,
                                                         broadcast_dims,
                                                         ellipsis_dims[1],
                                                         ops[1],
                                                         dB);
    }
  }
}
}  // namespace phi
