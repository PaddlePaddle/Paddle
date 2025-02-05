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
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace test {

pir::AttributeMap CreateAttributeMap(
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::AttributeMap attr_map;
  for (size_t i = 0; i < attribute_names.size(); i++) {
    pir::Attribute attr_value = pir::StrAttribute::get(ctx, attributes[i]);
    attr_map.insert(
        std::pair<std::string, pir::Attribute>(attribute_names[i], attr_value));
  }
  return attr_map;
}

pir::Operation *CreateDenseTensorOp(
    pir::IrContext *ctx,
    const phi::DDim &dims,
    const std::vector<std::string> &attribute_names,
    const std::vector<std::string> &attributes,
    const pir::Type &dtype =
        pir::Float32Type::get(pir::IrContext::Instance())) {
  std::vector<pir::Value> op_inputs = {};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  std::vector<pir::Type> op_output_types = {
      paddle::dialect::DenseTensorType::get(
          ctx, dtype, dims, data_layout, lod, offset)};

  return pir::Operation::Create(op_inputs,
                                CreateAttributeMap(attribute_names, attributes),
                                op_output_types,
                                pir::OpInfo());
}

}  // namespace test
