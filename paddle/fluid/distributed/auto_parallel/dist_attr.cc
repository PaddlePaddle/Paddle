/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"

#include <iostream>

namespace paddle {
namespace distributed {
namespace auto_parallel {

TensorDistributedAttribute::TensorDistributedAttribute(
    const VarDesc& tensor_desc)
    : tensor_desc_(&tensor_desc), batch_dim_(0) {
  set_default_dims_mapping();
  std::vector<int64_t> tensor_shape = tensor_desc_->GetShape();
  for (std::size_t i = 0; i < tensor_shape.size(); ++i) {
    dynamic_dims_.push_back(false);
  }
}

std::string TensorDistributedAttribute::to_string() const {
  std::string dist_str = "{tensor_name:" + tensor_desc_->Name() + ",";
  dist_str += "process_mesh:" + process_mesh_.to_string() + ",";
  dist_str += "dims_mappings:[" + str_join(dims_mapping_) + "],";
  dist_str += "batch_dim:" + std::to_string(batch_dim_) + ",";
  dist_str += "dynamic_dims:[" + str_join(dynamic_dims_) + "]}";
  return dist_str;
}

OperatorDistributedAttribute::OperatorDistributedAttribute(
    const OpDesc& op_desc)
    : op_desc_(&op_desc) {
  for (std::string name : op_desc_->InputArgumentNames()) {
    VarDesc* input = op_desc_->Block()->FindVarRecursive(name);
    inputs_[name] = input;
    input_dist_attrs_[name] = TensorDistributedAttribute(*input);
  }
  for (std::string name : op_desc_->OutputArgumentNames()) {
    VarDesc* output = op_desc_->Block()->FindVarRecursive(name);
    outputs_[name] = output;
    output_dist_attrs_[name] = TensorDistributedAttribute(*output);
  }
}

std::string OperatorDistributedAttribute::to_string() const {
  std::string dist_str = "{op_type:" + op_desc_->Type() + ",";
  dist_str += "input_dist_attrs:[";
  for (auto const& item : input_dist_attrs_) {
    dist_str += item.second.to_string() + ",";
  }
  dist_str.replace(dist_str.size() - 1, 1, "]");
  dist_str += ",output_dist_attrs:[";
  for (auto const& item : output_dist_attrs_) {
    dist_str += item.second.to_string() + ",";
  }
  dist_str.replace(dist_str.size() - 1, 1, "]");
  dist_str += ",impl_type:" + impl_type_ + ",";
  dist_str += "impl_idx:" + std::to_string(impl_idx_) + "}";
  return dist_str;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
