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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kAccuracy = "accuracy";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AccuracyEquivalenceTrans) {
  auto *op = node->Op();
  auto indices_op = *(map_inputs["Indices"].at(0));
  builder::Op out_op = *(map_inputs["Out"].at(0));
  auto label_op = *(map_inputs["Label"].at(0));
  std::vector<builder::Op> outputs;
  int64_t num_samples = out_op.GetType().GetShape()[0];

  auto total_data = builder::FullLike(
      indices_op, num_samples, indices_op.GetType().GetPrimitiveType(), {});
  int64_t dims = static_cast<int64_t>(indices_op.GetType().GetRank());
  std::vector<int64_t> broadcast_dimensions(dims, 0);
  std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);

  auto label_broad = builder::BroadcastInDim(
      label_op, broadcast_dimensions, indices_op.GetType());
  auto correct_data = builder::Equal(indices_op, label_broad);
  correct_data = builder::Convert(correct_data,
                                  {correct_data.GetType().GetShape(),
                                   indices_op.GetType().GetPrimitiveType()});

  std::vector<int64_t> perm(label_op.GetType().GetRank(), 0);
  std::iota(perm.begin(), perm.end(), 0);
  auto correct_num =
      builder::ReduceSum(correct_data, false, perm, total_data.GetType());

  auto total_data_f =
      builder::Convert(total_data, {{}, out_op.GetType().GetPrimitiveType()});
  auto correct_num_f =
      builder::Convert(correct_num, {{}, out_op.GetType().GetPrimitiveType()});
  auto accuracy_data = builder::Div(correct_num_f, total_data_f);

  outputs.push_back(accuracy_data);
  outputs.push_back(correct_num);
  outputs.push_back(total_data);

  std::vector<std::string> output_names{"Accuracy", "Correct", "Total"};
  auto output_name_map = op->Outputs();
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  builder::Op result = builder::Tuple(outputs);
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kAccuracy, INSENSITIVE, AccuracyEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
