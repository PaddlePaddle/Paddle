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

#include "paddle/fluid/ir/dialect/kernel_dialect.h"
#include "paddle/fluid/ir/dialect/kernel_op.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/pass/pd_op_to_kernel_pass.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {
namespace dialect {

phi::DataType convetIrType2DataType(ir::Type type) {
  if (type.isa<ir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (type.isa<ir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (type.isa<ir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (type.isa<ir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (type.isa<ir::Int32Type>()) {
    return phi::DataType::INT32;
  } else {
    PADDLE_THROW("not shupport type for now", type);
  }
}

phi::KernelKey GetKernelKey(
    ir::Operation* op,
    const phi::Place& place,
    const std::unordered_map<ir::Value, ir::OpResult>& map_value_pair) {
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
  auto op_info_res = op_info_interface.GetOpInfo();

  auto input_info = std::get<0>(op_info_res);

  // only suppurt non vector input for now
  std::map<std::string, int> input_map;
  int index = 0;
  for (auto& t : input_info) {
    // todo filter attribute tensor
    input_map[t.name] = index++;
  }

  std::cerr << "11" << std::endl;
  std::map<std::string, std::string> attr_type_map;
  auto attr_info = std::get<1>(op_info_res);
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }
  auto runtime_info = std::get<3>(op_info_res);

  std::cerr << "12" << std::endl;
  // get dtype infomation
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;

  auto attr_map = op->attributes();
  auto data_type_info = runtime_info.kernel_key_dtype;
  if (data_type_info.size() > 0 && data_type_info[0] != "") {
    // only support single input and attribute
    auto slot_name = data_type_info[0];
    if (input_map.count(slot_name)) {
      // parse from input
      int in_index = input_map.at(slot_name);

      dialect::AllocatedDenseTensorType type =
          op->GetOperandByIndex(in_index)
              .source()
              .type()
              .dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
      kernel_data_type = type.dyn_cast<dialect::DataTypeAttribute>().data();
    } else {
      PADDLE_ENFORCE_EQ(attr_type_map.count(slot_name),
                        true,
                        "[%s] MUST in attr map",
                        slot_name);
      kernel_data_type = attr_map.at(slot_name)
                             .dyn_cast<paddle::dialect::DataTypeAttribute>()
                             .data();
    }
  }

  std::cerr << "13" << std::endl;
  // parse all the input tensor

  std::cerr << "input size " << input_map.size() << std::endl;
  if (input_map.size() == 0 || op->name() == "pd.full_") {
    // all the information have to get from attribute and context
    kernel_backend = paddle::experimental::ParseBackend(place);

  } else {
    std::cerr << "1.1" << std::endl;
    paddle::experimental::detail::KernelKeyParser kernel_key_parser;

    for (size_t i = 0; i < input_info.size(); ++i) {
      // todo filter attribute tensor
      std::cerr << "1.1.0.1" << std::endl;
      auto input_tmp = op->GetOperandByIndex(i).source();
      auto new_input_tmp = map_value_pair.at(input_tmp);
      dialect::AllocatedDenseTensorType type =
          new_input_tmp.type().dyn_cast<dialect::AllocatedDenseTensorType>();
      std::cerr << "1.1.0.2" << std::endl;
      // fake tensor here
      phi::Place cpu_place(phi::AllocationType::CPU);
      auto ptr = new phi::Allocation(nullptr, 0, cpu_place);
      std::cerr << "1.1.0.3" << std::endl;
      std::shared_ptr<phi::Allocation> holder(ptr);
      std::cerr << "1.1.0" << std::endl;

      auto dtype = convetIrType2DataType(type.dtype());
      std::cerr << "dtype " << dtype << std::endl;
      phi::DenseTensorMeta meta(
          dtype, type.dims(), type.data_layout(), type.lod(), type.offset());
      std::cerr << "1.1.2" << std::endl;
      phi::DenseTensor fake_tensor(holder, meta);

      std::cerr << "1.1.1" << std::endl;
      kernel_key_parser.AssignKernelKeySet(fake_tensor);
    }

    std::cerr << "1.2" << std::endl;
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
  }

  std::cerr << "find res " << kernel_backend << "\t" << kernel_layout << "\t"
            << kernel_data_type << std::endl;
  phi::KernelKey res(kernel_backend, kernel_layout, kernel_data_type);
  return res;
}

std::unique_ptr<ir::Program> PdOpLowerToKernelPass(ir::Program* prog) {
  auto program = std::make_unique<ir::Program>(ir::IrContext::Instance());

  prog->Print(std::cout);
  auto block = prog->block();
  phi::Place cpu_place(phi::AllocationType::CPU);

  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleKernelDialect>();

  std::unordered_map<ir::Operation*, ir::Operation*> map_op_pair;
  std::unordered_map<ir::Value, ir::OpResult> map_value_pair;

  std::string op1_name = paddle::dialect::PhiKernelOp::name();

  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);

  std::unordered_map<std::string, ir::Attribute> op1_attribute{
      {"parameter_name", ir::StrAttribute::get(ctx, "a")}};

  for (auto it = block->begin(); it != block->end(); ++it) {
    std::cerr << (*it)->name() << std::endl;

    auto kernel_key = GetKernelKey(*it, cpu_place, map_value_pair);

    // create new Op

    // only for single output
    auto allocated_dense_tensor_dtype =
        paddle::dialect::AllocatedDenseTensorType::get(
            ctx,
            phi::TransToPhiPlace(kernel_key.backend()),
            (*it)
                ->GetResultByIndex(0)
                .type()
                .dyn_cast<dialect::DenseTensorType>());

    // constuct input
    std::vector<ir::OpResult> vec_inputs;
    if ((*it)->name() != "pd.full_" && (*it)->num_operands() > 0) {
      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        auto cur_in = (*it)->GetOperandByIndex(i).source();
        auto new_in = map_value_pair.at(cur_in);

        vec_inputs.push_back(new_in);
      }
    }

    ir::Operation* op1 = ir::Operation::Create(
        vec_inputs, op1_attribute, {allocated_dense_tensor_dtype}, op1_info);

    map_op_pair[*it] = op1;
    map_value_pair[(*it)->GetResultByIndex(0)] = op1->GetResultByIndex(0);

    program->block()->push_back(op1);
  }

  program->Print(std::cout);
  return program;
}

}  // namespace dialect
}  // namespace paddle
