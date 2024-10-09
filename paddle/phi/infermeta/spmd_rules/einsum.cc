/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <numeric>
#include <set>

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/einsum.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"
#include "paddle/utils/string/string_helper.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;
void ParseEinsumEquation(const std::string& equation,
                         std::vector<std::string>* operands,
                         std::string* output) {
  auto results = paddle::string::split_string(equation, "->");
  auto left = results[0];
  *operands = paddle::string::split_string(left, ",");
  *output = results[1];
}

bool IsEinsumOuter(const std::string& equation) {
  std::vector<std::string> inputs;
  std::string output;
  ParseEinsumEquation(equation, &inputs, &output);

  if (inputs.size() != 2) {
    return false;
  }

  std::unordered_map<char, int> input_char_count;
  for (const auto& in : inputs) {
    for (char c : in) {
      input_char_count[c]++;
      if (input_char_count[c] > 1) {
        return false;
      }
    }
  }

  std::unordered_map<char, int> output_char_count;
  for (char c : output) {
    output_char_count[c]++;
  }
  if (input_char_count != output_char_count) {
    return false;
  }
  return true;
}

// SpmdInfo EinsumInferSpmdBase(const std::vector<DistMetaTensor>& inputs) {
//   auto ori_x_shape = common::vectorize(inputs[0].dims());
//   auto ori_y_shape = common::vectorize(inputs[1].dims());
//   const auto& x_dist_attr_src = inputs[0].dist_attr();
//   const auto& y_dist_attr_src = inputs[1].dist_attr();
//   std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
//   std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
//   VLOG(6) << "EinsumInferSpmd InferForward Inputs: "
//           << "X shape: [" << str_join(ori_x_shape) << "], x_dims_mapping: ["
//           << str_join(x_dims_mapping) << "]; Y shape: ["
//           << str_join(ori_y_shape) << "], y_dims_mapping: ["
//           << str_join(y_dims_mapping);

//   // Step1: build Einsum Notation
//   auto results = paddle::string::split_string(equation, "->");
//   auto left = results[0];
//   auto operands = paddle::string::split_string(left, ",");
//   std::string x_axes = operands[0];
//   std::string y_axes = operands[1];
//   std::string out_axes = results[1];

//   // Step2: Sharding Propagation
//   std::unordered_map<char, int64_t> axis_to_dim_map;
//   for (size_t i = 0; i < x_axes.size(); ++i) {
//     axis_to_dim_map[x_axes[i]] = x_dims_mapping[i];
//   }
//   for (size_t i = 0; i < y_axes.size(); ++i) {
//     axis_to_dim_map[y_axes[i]] = y_dims_mapping[i];
//   }

//   std::vector<int64_t> out_dims_mapping;
//   out_dims_mapping.reserve(out_axes.size());

//   for (size_t i = 0; i < out_axes.size(); ++i) {
//     out_dims_mapping.push_back(axis_to_dim_map[out_axes[i]]);
//   }
//   TensorDistAttr x_dist_attr_dst =
//   CopyTensorDistAttrForOutput(x_dist_attr_src); TensorDistAttr
//   y_dist_attr_dst = CopyTensorDistAttrForOutput(y_dist_attr_src);
//   TensorDistAttr out_dist_attr_dst =
//       CopyTensorDistAttrForOutput(x_dist_attr_src);
//   out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

//   VLOG(4) << "EinsumSPMDRule InferForward: "
//           << "Einsum notation: [" << x_axes << "," << y_axes << " --> "
//           << out_axes << "]. " << std::endl;
//   LogInputDistAttr("X", ori_x_shape, x_dist_attr_src, x_dist_attr_dst);
//   LogInputDistAttr("Y", ori_y_shape, y_dist_attr_src, y_dist_attr_dst);
//   LogOutputDistAttr("Output", out_dist_attr_dst);

//   std::vector<TensorDistAttr> input_dist_attrs;
//   input_dist_attrs.push_back(x_dist_attr_dst);
//   input_dist_attrs.push_back(y_dist_attr_dst);

//   return {{input_dist_attrs}, {out_dist_attr_dst}};
// }

SpmdInfo EinsumInferSpmdBase(const std::vector<DistMetaTensor>& inputs,
                             bool is_outer,
                             int32_t out_dim) {
  PADDLE_ENFORCE_LE(
      inputs.size(),
      2,
      common::errors::InvalidArgument(
          "EinsumOp only support len(operands) between (0, 2]. Use "
          "opt_einsum first to convert multi-variable to binary-variable."));
  // case 1: not outer
  if (!is_outer || inputs.size() == 1) {
    PADDLE_ENFORCE_GT(
        out_dim,
        0,
        common::errors::InvalidArgument("out_dim should be greater than 0 when "
                                        "is_outer is false, but received %d",
                                        out_dim));

    std::vector<TensorDistAttr> input_dist_attrs_dst;
    for (auto& input : inputs) {
      TensorDistAttr replicated_dist_attr =
          GetReplicatedDistAttr(input.dist_attr());
      input_dist_attrs_dst.push_back(replicated_dist_attr);
    }

    std::vector<int64_t> fake_output_shape(out_dim, 1);
    TensorDistAttr out_dist_attr_dst(fake_output_shape);
    return {{input_dist_attrs_dst}, {out_dist_attr_dst}};
  }

  auto ori_x_shape = common::vectorize(inputs[0].dims());
  auto ori_y_shape = common::vectorize(inputs[1].dims());
  const auto& x_dist_attr_src = inputs[0].dist_attr();
  const auto& y_dist_attr_src = inputs[1].dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  VLOG(0) << "EinsumInferSpmd InferForward Inputs: "
          << "X shape: [" << str_join(ori_x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; Y shape: ["
          << str_join(ori_y_shape) << "], y_dims_mapping: ["
          << str_join(y_dims_mapping) << "]";

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(y_dist_attr_src);
  // case 2: outer but inputs are both sharded
  // TODO(dev): consider the case when output is transposed
  if (x_dist_attr_src.is_shard() && y_dist_attr_src.is_shard()) {
    x_dist_attr_dst = GetReplicatedDistAttr(x_dist_attr_src);
  } else {
    x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  }

  y_dist_attr_dst.set_dims_mapping(y_dims_mapping);
  std::vector<TensorDistAttr> input_dist_attrs;
  input_dist_attrs.push_back(x_dist_attr_dst);
  input_dist_attrs.push_back(y_dist_attr_dst);

  std::vector<int64_t> out_dims_mapping;
  out_dims_mapping.reserve(x_dims_mapping.size() + y_dims_mapping.size());

  for (size_t i = 0; i < x_dims_mapping.size(); ++i) {
    out_dims_mapping.push_back(x_dist_attr_dst.dims_mapping()[i]);
  }

  for (size_t i = 0; i < y_dims_mapping.size(); ++i) {
    out_dims_mapping.push_back(y_dist_attr_dst.dims_mapping()[i]);
  }
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  LogInputDistAttr("X", ori_x_shape, x_dist_attr_src, x_dist_attr_dst);
  LogInputDistAttr("Y", ori_y_shape, y_dist_attr_src, y_dist_attr_dst);
  LogOutputDistAttr("Output", out_dist_attr_dst);
  VLOG(4) << std::endl;
  return {{input_dist_attrs}, {out_dist_attr_dst}};
}

SpmdInfo EinsumInferSpmd(const std::vector<DistMetaTensor>& inputs,
                         const std::string& equation) {
  // Step0: verify input args based on einsum logic
  PADDLE_ENFORCE_LE(
      inputs.size(),
      2,
      common::errors::InvalidArgument(
          "EinsumOp only support len(operands) between (0, 2]. Use "
          "opt_einsum first to convert multi-variable to binary-variable."));

  std::vector<std::string> operands;
  std::string right;
  ParseEinsumEquation(equation, &operands, &right);
  return EinsumInferSpmdBase(inputs, IsEinsumOuter(equation), right.size());
}

SpmdInfo EinsumGradInferSpmdBase(const std::vector<DistMetaTensor>& inputs,
                                 const DistMetaTensor& out_grad,
                                 bool is_outer) {
  PADDLE_ENFORCE_LE(
      inputs.size(),
      2,
      common::errors::InvalidArgument(
          "EinsumOp only support len(operands) between (0, 2]. Use "
          "opt_einsum first to convert multi-variable to binary-variable."));

  if (!is_outer || inputs.size() == 1) {
    PADDLE_ENFORCE_GT(
        out_dim,
        0,
        common::errors::InvalidArgument("out_dim should be greater than 0 when "
                                        "is_outer is false, but received %d",
                                        out_dim));

    std::vector<TensorDistAttr> input_dist_attrs_dst;
    for (auto& input : inputs) {
      TensorDistAttr replicated_dist_attr =
          GetReplicatedDistAttr(input.dist_attr());
      input_dist_attrs_dst.push_back(replicated_dist_attr);
    }

    TensorDistAttr out_grad_dist_attr_dst =
        GetReplicatedDistAttr(out_grad.dist_attr());
    return {{input_dist_attrs_dst, out_grad.dist_attr()},
            {input_dist_attrs_dst}};
  }

  auto ori_x_shape = common::vectorize(inputs[0].dims());
  auto ori_y_shape = common::vectorize(inputs[1].dims());
  auto ori_out_grad_shape = common::vectorize(out_grad.dims());
  const auto& x_dist_attr_src = inputs[0].dist_attr();
  const auto& y_dist_attr_src = inputs[1].dist_attr();
  const auto& out_grad_dist_attr_src = out_grad.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  std::vector<int64_t> out_grad_dims_mapping =
      out_grad_dist_attr_src.dims_mapping();
  VLOG(0) << "EinsumGradInferSpmd Inputs: "
          << "X shape: [" << str_join(ori_x_shape) << "], x_dims_mapping: ["
          << str_join(x_dims_mapping) << "]; Y shape: ["
          << str_join(ori_y_shape) << "], y_dims_mapping: ["
          << str_join(y_dims_mapping) << "]; Out_grad shape: ["
          << str_join(ori_out_grad_shape) << "], out_grad_dims_mapping: ["
          << str_join(out_grad_dims_mapping) << "]";

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr y_dist_attr_dst = CopyTensorDistAttrForOutput(y_dist_attr_src);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  TensorDistAttr x_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr y_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(y_dist_attr_src);
  if (out_grad_dist_attr_src.is_shard()) {
    x_dist_attr_dst = GetReplicatedDistAttr(x_dist_attr_src);
    y_dist_attr_dst = GetReplicatedDistAttr(y_dist_attr_src);

    // 情况1: out grad 的 shard
    std::vector<int64_t> x_grad_partial_on_dims;
    for (size_t i = 0; i < ori_x_shape.size(); ++i) {
      if (out_grad_dims_mapping[i] != -1 && x_dims_mapping[i] != -1) {
        x_grad_partial_on_dims.push_back(i);
      }
    }
    x_grad_dist_attr_dst.set_partial_status(x_grad_partial_on_dims);

    std::vector<int64_t> y_grad_partial_on_dims;
    for (size_t i = 0; i < ori_y_shape.size(); ++i) {
      auto j = ori_x_shape.size() + i;
      if (out_grad_dims_mapping[j] != -1 && y_dims_mapping[i] != -1) {
        y_grad_partial_on_dims.push_back(i);
      }
    }
    y_grad_dist_attr_dst.set_partial_status(y_grad_partial_on_dims);
  } else {
    x_dist_attr_dst = GetReplicatedDistAttr(x_dist_attr_src);
    y_dist_attr_dst = GetReplicatedDistAttr(y_dist_attr_src);
    out_grad_dist_attr_dst = GetReplicatedDistAttr(out_grad_dist_attr_src);
  }
}

SpmdInfo EinsumGradInferSpmd(const std::vector<DistMetaTensor>& inputs,
                             const DistMetaTensor& out_grad,
                             const std::string& equation) {
  PADDLE_ENFORCE_LE(
      inputs.size(),
      2,
      common::errors::InvalidArgument(
          "EinsumOp only support len(operands) between (0, 2]. Use "
          "opt_einsum first to convert multi-variable to binary-variable."));

  std::vector<std::string> operands;
  std::string right;
  ParseEinsumEquation(equation, &operands, &right);
  return EinsumGradInferSpmdBase(inputs, out_grad, IsEinsumOuter(equation));
}

}  // namespace phi::distributed
