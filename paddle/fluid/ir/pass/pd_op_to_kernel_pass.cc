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

#include <iostream>

#include "paddle/fluid/ir/pass/pd_op_to_kernel_pass.h"

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_dialect.h"
#include "paddle/fluid/ir/dialect/kernel_op.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
namespace paddle {
namespace dialect {

phi::KernelKey GetKernelKey(
    ir::Operation* op,
    const phi::Place& place,
    const std::unordered_map<ir::Value, ir::OpResult>& map_value_pair) {
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
  auto op_info_res = op_info_interface.GetOpInfo();

  auto input_info = std::get<0>(op_info_res);

  std::cerr << "get kernel key" << std::endl;
  // only suppurt non vector input for now
  std::map<std::string, int> input_map;
  int index = 0;
  for (auto& t : input_info) {
    // todo filter attribute tensor
    input_map[t.name] = index++;
  }

  std::cerr << "111" << std::endl;
  std::map<std::string, std::string> attr_type_map;
  auto attr_info = std::get<1>(op_info_res);
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }
  auto runtime_info = std::get<3>(op_info_res);

  std::cerr << "112" << std::endl;
  // get dtype infomation
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;

  auto attr_map = op->attributes();
  auto data_type_info = runtime_info.kernel_key_dtype;
  if (data_type_info.size() > 0 && data_type_info[0] != "") {
    // only support single input and attribute
    auto slot_name = data_type_info[0];
    std::cerr << "slot name " << slot_name << std::endl;
    if (input_map.count(slot_name)) {
      // parse from input
      std::cerr << "in slot " << std::endl;
      int in_index = input_map.at(slot_name);

      std::cerr << "in 2 " << in_index << std::endl;
      dialect::DenseTensorType type =
          op->operand(in_index)
              .source()
              .type()
              .dyn_cast<paddle::dialect::DenseTensorType>();
      std::cerr << "in 3 " << std::endl;
      kernel_data_type = TransToPhiDataType(type.dtype());
      std::cerr << "in 4 " << std::endl;
    } else {
      PADDLE_ENFORCE_EQ(
          attr_type_map.count(slot_name),
          true,
          phi::errors::PreconditionNotMet("[%s] MUST in attr map", slot_name));
      kernel_data_type = attr_map.at(slot_name)
                             .dyn_cast<paddle::dialect::DataTypeAttribute>()
                             .data();
    }
  }

  std::cerr << "112 1" << std::endl;
  // parse all the input tensor

  if (input_map.size() == 0 || op->name() == "pd.full_") {
    // all the information have to get from attribute and context
    kernel_backend = paddle::experimental::ParseBackend(place);

  } else {
    paddle::experimental::detail::KernelKeyParser kernel_key_parser;

    for (size_t i = 0; i < input_info.size(); ++i) {
      std::cerr << "int i " << i << std::endl;
      // todo filter attribute tensor
      auto input_tmp = op->operand(i).source();
      auto new_input_tmp = map_value_pair.at(input_tmp);
      std::cerr << "113" << std::endl;
      dialect::AllocatedDenseTensorType type =
          new_input_tmp.type().dyn_cast<dialect::AllocatedDenseTensorType>();

      // fake tensor here
      auto ptr = new phi::Allocation(nullptr, 0, type.place());

      std::shared_ptr<phi::Allocation> holder(ptr);

      auto dtype = TransToPhiDataType(type.dtype());
      std::cerr << "115" << std::endl;
      phi::DenseTensorMeta meta(
          dtype, type.dims(), type.data_layout(), type.lod(), type.offset());

      phi::DenseTensor fake_tensor(holder, meta);

      kernel_key_parser.AssignKernelKeySet(fake_tensor);

      std::cerr << "116" << std::endl;
    }

    auto kernel_key_set = kernel_key_parser.key_set;

    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

    if (kernel_backend == phi::Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == phi::DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == phi::DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
    std::cerr << "118" << std::endl;
  }

  phi::KernelKey res(kernel_backend, kernel_layout, kernel_data_type);
  return res;
}

std::unique_ptr<ir::Program> PdOpLowerToKernelPass(ir::Program* prog) {
  auto program = std::make_unique<ir::Program>(ir::IrContext::Instance());

  auto block = prog->block();
  phi::Place cpu_place(phi::AllocationType::CPU);

  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleKernelDialect>();

  std::unordered_map<ir::Operation*, ir::Operation*> map_op_pair;
  std::unordered_map<ir::Value, ir::OpResult> map_value_pair;

  std::string op1_name = paddle::dialect::PhiKernelOp::name();

  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);

  for (auto it = block->begin(); it != block->end(); ++it) {
    std::cerr << "begin " << (*it)->name() << std::endl;
    auto kernel_key = GetKernelKey(*it, cpu_place, map_value_pair);

    // create new Op

    // only for single output
    // need update new kernel key layout and data tyep

    std::vector<ir::Type> op_output_types;
    if ((*it)->num_results() > 0) {
      auto allocated_dense_tensor_dtype =
          paddle::dialect::AllocatedDenseTensorType::get(
              ctx,
              phi::TransToPhiPlace(kernel_key.backend()),
              (*it)->result(0).type().dyn_cast<dialect::DenseTensorType>());
      op_output_types.push_back(allocated_dense_tensor_dtype);
    }
    // constuct input
    std::vector<ir::OpResult> vec_inputs;
    if ((*it)->name() != "pd.full_" && (*it)->num_operands() > 0) {
      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        auto cur_in = (*it)->operand(i).source();
        auto new_in = map_value_pair.at(cur_in);

        vec_inputs.push_back(new_in);
      }
    }
    std::cerr << "11  " << std::endl;
    paddle::dialect::OpYamlInfoInterface op_info_interface =
        (*it)->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
    auto op_info_res = op_info_interface.GetOpInfo();
    auto runtime_info = std::get<3>(op_info_res);

    std::cerr << "kernel name " << runtime_info.kernel_func[0] << std::endl;
    std::unordered_map<std::string, ir::Attribute> op1_attribute{
        {"op_name", ir::StrAttribute::get(ctx, (*it)->name())},
        {"kernel_name",
         ir::StrAttribute::get(ctx, runtime_info.kernel_func[0])},
        {"kernel_key", dialect::KernelAttribute::get(ctx, kernel_key)}};

    auto op_attr_map = (*it)->attributes();
    std::cerr << "112 " << std::endl;

    for (auto it1 = op_attr_map.begin(); it1 != op_attr_map.end(); ++it1) {
      op1_attribute.emplace(it1->first, it1->second);
    }
    std::cerr << "13  " << std::endl;
    ir::Operation* op1 = ir::Operation::Create(
        vec_inputs, op1_attribute, op_output_types, op1_info);

    map_op_pair[*it] = op1;

    // only deal with single output
    if ((*it)->num_results() > 0) {
      map_value_pair[(*it)->result(0)] = op1->result(0);
    }

    program->block()->push_back(op1);
  }

  return program;
}

}  // namespace dialect
}  // namespace paddle
