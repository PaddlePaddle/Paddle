/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/einsum.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
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

void ConstraintOnDiagLabel(std::vector<std::string>* operands,
                           std::string* output) {
  // Empirically, for fwd calculation, only those diagonal labels in output
  // should not be sharded. e.g. iji->ii (diag), 'i' cannot be sharded;
  // e.g. iji->i (trace), 'i' can be sharded.
  // But during bwd calculation, input and output are switched.
  // e.g. in the 'trace' case above when calculating x_grad, it will use
  // i->ii, so 'i' cannot be sharded.
  // Thus we simply set the spmd rule here to replace all diagonal labels as 1.

  // find diagonal labels
  std::unordered_map<char, int> char_count;
  std::unordered_set<char> diagonal_labels;
  for (auto op : *operands) {
    for (char c : op) {
      char_count[c]++;
      if (char_count[c] > 1) {
        diagonal_labels.insert(c);
      }
    }
    char_count.clear();
  }
  for (char c : *output) {
    char_count[c]++;
    if (char_count[c] > 1) {
      diagonal_labels.insert(c);
    }
  }

  if (diagonal_labels.size()) {
    // replace input operands' diagonal labels
    for (size_t i = 0; i < operands->size(); ++i) {
      for (size_t j = 0; j < (*operands)[i].size(); ++j) {
        if (diagonal_labels.find((*operands)[i][j]) != diagonal_labels.end()) {
          (*operands)[i].replace(j, 1, "1");
        }
      }
    }
    // replace output's diagonal labels
    for (size_t i = 0; i < output->size(); ++i) {
      if (diagonal_labels.find((*output)[i]) != diagonal_labels.end()) {
        output->replace(i, 1, "1");
      }
    }
  }
}

bool IsEinsumOuter(const std::vector<std::string>& inputs,
                   const std::string& output) {
  // Outer case: e.g. i, j -> ij; ij, kl -> ijkl
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

void ConstraintOnOuter(const phi::distributed::TensorDistAttr& x_attr,
                       const phi::distributed::TensorDistAttr& y_attr,
                       int x_ndim,
                       int y_ndim,
                       std::vector<int64_t>* x_dims_mapping,
                       std::vector<int64_t>* y_dims_mapping) {
  // For outer operation, only one operand and one dimension can be sharded
  // todo: if multiple dimensions are requested to be sharded, decide which
  // operand and which dimension to be sharded could be better

  // we simply choose the first operand requested to be sharded and the
  // first dimension requested to be sharded here
  if (x_attr.is_shard()) {
    bool meet_shard_axis = false;
    for (int i = 0; i < x_ndim; ++i) {
      if ((*x_dims_mapping)[i] != -1) {
        meet_shard_axis = true;
        continue;
      }
      if (meet_shard_axis) {
        (*x_dims_mapping)[i] = -1;
      }
    }
    // reset y_dims_mapping to all replicated
    for (int i = 0; i < y_ndim; ++i) {
      (*y_dims_mapping)[i] = -1;
    }
  } else if (y_attr.is_shard()) {
    bool meet_shard_axis = false;
    for (int i = 0; i < y_ndim; ++i) {
      if ((*y_dims_mapping)[i] != -1) {
        meet_shard_axis = true;
        continue;
      }
      if (meet_shard_axis) {
        (*y_dims_mapping)[i] = -1;
      }
    }
    // no need to reset x_dims_mapping
  }
}

SpmdInfo EinsumInferSpmd(const std::vector<DistMetaTensor>& inputs,
                         const std::string& equation) {
  PADDLE_ENFORCE_LE(
      inputs.size(),
      2,
      common::errors::InvalidArgument(
          "EinsumOp only support len(operands) between (0, 2]. Use "
          "opt_einsum first to convert multi-variable to binary-variable."));

  std::vector<std::string> operands;
  std::string right;
  // ellipsis labels are already parsed in python API (einsum_v2)
  ParseEinsumEquation(equation, &operands, &right);
  // diagonal case
  ConstraintOnDiagLabel(&operands, &right);

  if (inputs.size() == 1) {
    // single operand
    DistMetaTensor x = inputs[0];
    EXTRACT_SHAPE_AND_DIST_ATTR(x);
    std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);

    VLOG(6) << "EinsumInferSpmd InferForward Inputs: "
            << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
            << str_join(x_dims_mapping);

    // Step1: Sharding Propagation
    // Step1.1: Merge input shardings
    std::unordered_map<std::string, int64_t> axis_to_dim_map =
        ShardingMergeForTensors({{operands[0], x_dims_mapping}});

    // Step1.2: Infer output dims mapping
    TensorDistAttr x_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    x_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[0], axis_to_dim_map));

    std::vector<int64_t> fake_output_shape(right.size(), 1);
    TensorDistAttr out_dist_attr_dst(fake_output_shape);
    out_dist_attr_dst.set_process_mesh(x_dist_attr_src.process_mesh());
    out_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(right, axis_to_dim_map));

    // Step2: Handle Partial
    // Step2.1 Output Partial
    std::vector<int64_t> partial_on_dims =
        ResoluteOutputPartialDimension(axis_to_dim_map, right);
    out_dist_attr_dst.set_partial_status(partial_on_dims);

    VLOG(4) << "x_axes: " << operands[0] << " out_axes: " << right;
    LOG_SPMD_INPUT(x);
    VLOG(4) << "out";
    VLOG(4) << "dist_attr: [" << out_dist_attr_dst.to_string() << "]";

    std::vector<TensorDistAttr> input_dist_attrs;
    input_dist_attrs.push_back(x_dist_attr_dst);
    return {{input_dist_attrs}, {out_dist_attr_dst}};
  } else {
    // double operands
    DistMetaTensor x = inputs[0];
    DistMetaTensor y = inputs[1];
    EXTRACT_SHAPE_AND_DIST_ATTR(x);
    EXTRACT_SHAPE_AND_DIST_ATTR(y);
    std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
    std::vector<int64_t> y_dims_mapping(y_dims_mapping_src);

    if (IsEinsumOuter(operands, right)) {
      ConstraintOnOuter(x_dist_attr_src,
                        y_dist_attr_src,
                        x_ndim,
                        y_ndim,
                        &x_dims_mapping,
                        &y_dims_mapping);
    }
    VLOG(6) << "EinsumInferSpmd InferForward Inputs: "
            << "X shape: [" << str_join(x_shape) << "], x_dims_mapping: ["
            << str_join(x_dims_mapping) << "], Y shape: [" << str_join(y_shape)
            << "], y_dims_mapping: [" << str_join(y_dims_mapping);

    // Step1: Sharding Propagation
    // Step1.1: Merge input shardings
    std::unordered_map<std::string, int64_t> axis_to_dim_map =
        ShardingMergeForTensors(
            {{operands[0], x_dims_mapping}, {operands[1], y_dims_mapping}});

    // Step1.2: Infer output dims mapping
    TensorDistAttr x_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    TensorDistAttr y_dist_attr_dst =
        CopyTensorDistAttrForOutput(y_dist_attr_src);
    x_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[0], axis_to_dim_map));
    y_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[1], axis_to_dim_map));

    std::vector<int64_t> fake_output_shape(right.size(), 1);
    TensorDistAttr out_dist_attr_dst(fake_output_shape);
    out_dist_attr_dst.set_process_mesh(x_dist_attr_src.process_mesh());
    out_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(right, axis_to_dim_map));

    // Step2: Handle Partial
    // Step2.1 Output Partial
    std::vector<int64_t> partial_on_dims =
        ResoluteOutputPartialDimension(axis_to_dim_map, right);
    out_dist_attr_dst.set_partial_status(partial_on_dims);

    VLOG(4) << "x_axes: " << operands[0] << " y_axes: " << operands[1]
            << " out_axes: " << right;
    LOG_SPMD_INPUT(x);
    LOG_SPMD_INPUT(y);
    VLOG(4) << "out";
    VLOG(4) << "dist_attr: [" << out_dist_attr_dst.to_string() << "]";

    std::vector<TensorDistAttr> input_dist_attrs;
    input_dist_attrs.push_back(x_dist_attr_dst);
    input_dist_attrs.push_back(y_dist_attr_dst);

    return {{input_dist_attrs}, {out_dist_attr_dst}};
  }
}

SpmdInfo EinsumGradInferSpmd(const std::vector<DistMetaTensor>& inputs,
                             const std::vector<DistMetaTensor>& inner_cache,
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
  // ellipsis labels are already parsed in python API (einsum_v2)
  ParseEinsumEquation(equation, &operands, &right);
  // diagonal case
  ConstraintOnDiagLabel(&operands, &right);

  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  if (inputs.size() == 1) {
    // single operand
    DistMetaTensor x = inputs[0];
    EXTRACT_SHAPE_AND_DIST_ATTR(x);

    // For reduce label type in equation "right->left" used in backward
    // calculation, the gradient on those axes are tiled and copied, so we can
    // just copy the dims_mapping on those axes from input to input_grad.
    // Therefore we also merge the input axes here.
    std::unordered_map<std::string, int64_t> axis_to_dim_map =
        ShardingMergeForTensors({{operands[0], x_dims_mapping_src},
                                 {right, out_grad_dims_mapping_src}});

    TensorDistAttr x_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    x_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[0], axis_to_dim_map));

    TensorDistAttr out_grad_dist_attr_dst(out_grad_dist_attr_src);
    out_grad_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(right, axis_to_dim_map));

    std::vector<TensorDistAttr> input_dist_attrs;
    input_dist_attrs.push_back(x_dist_attr_dst);
    return {{input_dist_attrs, out_grad_dist_attr_dst}, {input_dist_attrs}};
  } else {
    // double operands
    DistMetaTensor x = inputs[0];
    DistMetaTensor y = inputs[1];
    EXTRACT_SHAPE_AND_DIST_ATTR(x);
    EXTRACT_SHAPE_AND_DIST_ATTR(y);
    std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
    std::vector<int64_t> y_dims_mapping(y_dims_mapping_src);
    std::vector<int64_t> out_grad_dims_mapping(out_grad_dims_mapping_src);

    if (IsEinsumOuter(operands, right)) {
      ConstraintOnOuter(x_dist_attr_src,
                        y_dist_attr_src,
                        x_ndim,
                        y_ndim,
                        &x_dims_mapping,
                        &y_dims_mapping);
    }
    // out_grad, x, y
    std::unordered_map<std::string, int64_t> fwd_axis_to_dim_map =
        ShardingMergeForTensors(
            {{operands[0], x_dims_mapping}, {operands[1], y_dims_mapping}});
    out_grad_dims_mapping = GetDimsMappingForAxes(right, fwd_axis_to_dim_map);
    TensorDistAttr out_grad_dist_attr_dst =
        CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
    out_grad_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(right, fwd_axis_to_dim_map));
    TensorDistAttr x_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    x_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[0], fwd_axis_to_dim_map));
    TensorDistAttr y_dist_attr_dst =
        CopyTensorDistAttrForOutput(y_dist_attr_src);
    y_dist_attr_dst.set_dims_mapping(
        GetDimsMappingForAxes(operands[1], fwd_axis_to_dim_map));

    // For reduce label type in equation "left[1], right->left[0]" and "right,
    // left[0]->left[1]" used in backward calculation, the gradient on those
    // axes are tiled and copied, so we can just copy the dims_mapping on those
    // axes from input to input_grad. Therefore we just copy the fwd inferred
    // input_dist_attr for input_grad_dist_attr and then handle partial.

    // dx = einsum(y, d_out)
    TensorDistAttr x_grad_dist_attr_dst = TensorDistAttr(x_dist_attr_dst);
    std::unordered_map<std::string, int64_t> axis_to_dim_map_for_dx =
        ShardingMergeForTensors(
            {{operands[1], y_dims_mapping}, {right, out_grad_dims_mapping}});
    // Handle Partial for dx
    std::vector<int64_t> partial_on_dx_dims =
        ResoluteOutputPartialDimension(axis_to_dim_map_for_dx, operands[0]);
    x_grad_dist_attr_dst.set_partial_status(partial_on_dx_dims);

    // dy = einsum(d_out, x)
    TensorDistAttr y_grad_dist_attr_dst = TensorDistAttr(y_dist_attr_dst);
    std::unordered_map<std::string, int64_t> axis_to_dim_map_for_dy =
        ShardingMergeForTensors(
            {{right, out_grad_dims_mapping}, {operands[0], x_dims_mapping}});
    // Handle Partial for dy
    std::vector<int64_t> partial_on_dy_dims =
        ResoluteOutputPartialDimension(axis_to_dim_map_for_dy, operands[1]);
    y_grad_dist_attr_dst.set_partial_status(partial_on_dy_dims);

    std::vector<TensorDistAttr> input_dist_attrs;
    input_dist_attrs.push_back(x_dist_attr_dst);
    input_dist_attrs.push_back(y_dist_attr_dst);
    std::vector<TensorDistAttr> input_grad_dist_attrs;
    input_grad_dist_attrs.push_back(x_grad_dist_attr_dst);
    input_grad_dist_attrs.push_back(y_grad_dist_attr_dst);
    return {{input_dist_attrs, out_grad_dist_attr_dst},
            {input_grad_dist_attrs}};
  }
}
}  // namespace phi::distributed
