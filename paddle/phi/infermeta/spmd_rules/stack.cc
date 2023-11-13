/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/stack.h"

#include <limits>
#include <set>

#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

std::string FillStackNotation(int64_t n_axis) {
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";
  PADDLE_ENFORCE_GT(alphabet.size(),
                    static_cast<size_t>(n_axis),
                    phi::errors::InvalidArgument(
                        "alphabet.size() [%d]; n_axis [%d] is too large",
                        alphabet.size(),
                        n_axis));
  std::string all_axis = alphabet.substr(0, n_axis);
  return all_axis;
}

SpmdInfo StackInferSpmd(const std::vector<DistMetaTensor>& x, int axis) {
  // 1、check tensors shapes
  std::vector<std::vector<int64_t>> tensor_shapes;
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(tensor_shapes),
                 [](const DistMetaTensor& meta) {
                   return phi::vectorize<int64_t>(meta.dims());
                 });
  bool all_empty =
      std::all_of(tensor_shapes.begin(), tensor_shapes.end(), IsEmpty);
  if (all_empty) {
    return SpmdInfo();
  }

  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });

  auto non_empty_index = non_empty_iter - tensor_shapes.begin();
  auto ndim = tensor_shapes[non_empty_index].size();
  // normlize dim
  auto dim = axis < 0 ? static_cast<int64_t>(ndim) + axis : axis;
  std::vector<TensorDistAttr> input_attrs;
  std::transform(
      x.begin(), x.end(), std::back_inserter(input_attrs), [](auto& meta) {
        return meta.dist_attr();
      });
  std::string notation = FillStackNotation(ndim);
  std::vector<std::string> axis_names(input_attrs.size(), notation);
  AlignDimsSharding(
      &input_attrs, tensor_shapes, axis_names, {}, notation, true);

  TensorDistAttr output_attr =
      CopyTensorDistAttrForOutput(input_attrs[non_empty_index], false);
  std::vector<int64_t> dim_mapping(ndim + 1, -1);
  const auto& input_dim_mapping = input_attrs[non_empty_index].dims_mapping();
  for (size_t i = 0; i < ndim; i++) {
    size_t out_index = i < static_cast<size_t>(dim) ? i : (i + 1);
    dim_mapping[out_index] = input_dim_mapping[i];
  }
  output_attr.set_dims_mapping(dim_mapping);
  return {{input_attrs}, {output_attr}};
}

SpmdInfo StackInferSpmdReverse(const std::vector<DistMetaTensor>& x,
                               const DistMetaTensor& output,
                               int axis) {
  auto out_dist_attr = output.dist_attr();
  out_dist_attr = UnShardTensorDim(out_dist_attr, axis);
  auto n_inputs = x.size();
  TensorDistAttr input_attr = CopyTensorDistAttrForOutput(out_dist_attr, false);
  auto ndim = output.dims().size();
  std::vector<int64_t> dim_mapping(ndim - 1, -1);
  const auto& input_dim_mapping = out_dist_attr.dims_mapping();
  for (size_t i = 0; i < static_cast<size_t>(ndim - 1); i++) {
    size_t out_index = i < static_cast<size_t>(axis) ? i : (i + 1);
    dim_mapping[i] = input_dim_mapping[out_index];
  }
  input_attr.set_dims_mapping(dim_mapping);
  std::vector<TensorDistAttr> input_attrs(n_inputs, input_attr);
  return {{input_attrs}, {output.dist_attr()}};
}

std::tuple<std::string, std::string> FillStackGradNotation(int64_t n_axis,
                                                           int64_t stack_dim) {
  static const std::string alphabet = "abcdefghijlopqrstuvwxyz";

  PADDLE_ENFORCE_GT(alphabet.size(),
                    static_cast<size_t>(n_axis),
                    phi::errors::InvalidArgument(
                        "alphabet.size() [%d]; n_axis [%d] is too large",
                        alphabet.size(),
                        n_axis));

  std::string input_axis = alphabet.substr(0, n_axis);
  std::string output_axis = input_axis.substr(0, stack_dim) +
                            alphabet[stack_dim] + input_axis.substr(stack_dim);
  return {input_axis, output_axis};
}

SpmdInfo StackGradInferSpmd(const DistMetaTensor& output_grad, int axis) {
  auto out_dist_attr = output_grad.dist_attr();
  out_dist_attr = UnShardTensorDim(out_dist_attr, axis);
  auto n_inputs = output_grad.dims().at(axis);
  TensorDistAttr input_attr = CopyTensorDistAttrForOutput(out_dist_attr, false);
  auto ndim = output_grad.dims().size();
  std::vector<int64_t> dim_mapping(ndim - 1, -1);
  const auto& input_dim_mapping = out_dist_attr.dims_mapping();
  for (size_t i = 0; i < static_cast<size_t>(ndim - 1); i++) {
    size_t out_index = i < static_cast<size_t>(axis) ? i : (i + 1);
    dim_mapping[i] = input_dim_mapping[out_index];
  }
  input_attr.set_dims_mapping(dim_mapping);
  std::vector<TensorDistAttr> input_attrs(n_inputs, input_attr);
  return {{out_dist_attr}, {input_attrs}};
}

}  // namespace distributed
}  // namespace phi
