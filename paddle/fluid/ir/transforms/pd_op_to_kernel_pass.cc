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

#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_dialect.h"
#include "paddle/fluid/ir/dialect/kernel_op.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/fluid/ir/trait/inplace.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
namespace paddle {
namespace dialect {

const int init_on_gpu_threashold = 1000;

std::unordered_map<std::string, phi::DataType> Str2PhiDataType = {
    {"DataType::FLOAT16", phi::DataType::FLOAT16},
    {"DataType::BFLOAT16", phi::DataType::BFLOAT16},
    {"DataType::FLOAT32", phi::DataType::FLOAT32},
    {"DataType::FLOAT64", phi::DataType::FLOAT64},
    {"DataType::INT16", phi::DataType::INT16},
    {"DataType::INT32", phi::DataType::INT32},
    {"DataType::INT64", phi::DataType::INT64},
    {"DataType::INT8", phi::DataType::INT8},
    {"DataType::BOOL", phi::DataType::BOOL},
};

const std::unordered_set<std::string> UnchangeOutputOps = {
    "pd.data",
    "builtin.combine",
    "builtin.slice",
    "pd.feed",
    "pd.fetch",
    "builtin.set_parameter",
    "builtin.get_parameter",
    "pd.shadow_output"};

const std::unordered_set<std::string> LegacyOpList = {
    "pd.fused_softmax_mask_upper_triangle",
    "pd.fused_softmax_mask_upper_triangle_grad"};

bool NeedFallBackCpu(const ir::Operation* op,
                     const std::string& kernel_fn_name,
                     const phi::KernelKey& kernel_key) {
  if (UnchangeOutputOps.count(op->name())) {
    return false;
  }
  if (kernel_fn_name == "") {
    return false;
  }
  if (phi::KernelFactory::Instance().HasKernel(kernel_fn_name, kernel_key)) {
    return false;
  }

  phi::KernelKey copy_kernel_key = kernel_key;
  if (copy_kernel_key.backend() == phi::Backend::GPUDNN) {
    copy_kernel_key.set_backend(phi::Backend::GPU);

    if (phi::KernelFactory::Instance().HasKernel(kernel_fn_name,
                                                 copy_kernel_key)) {
      return false;
    }
  }

  copy_kernel_key.set_backend(phi::Backend::CPU);
  if (phi::KernelFactory::Instance().HasKernel(kernel_fn_name,
                                               copy_kernel_key)) {
    return true;
  }

  return false;
}

bool NeedFallBackFromGPUDNN2GPU(ir::Operation* op,
                                const phi::KernelKey kernel_key) {
  // NOTE(phlrain): keep the same kernel select strategy with
  // GetExepectKernelKey
  if (op->name() == "pd.pool2d" || op->name() == "pd.pool2d_grad") {
    if (kernel_key.backend() == phi::Backend::GPUDNN &&
        (op->attributes().at("adaptive").dyn_cast<ir::BoolAttribute>().data() ==
         true)) {
      return true;
    }
  }

  return false;
}

ir::OpResult AddPlaceTransferOp(ir::OpResult in,
                                ir::Type out_type,
                                const phi::Place& src_place,
                                const phi::Place& dst_place,
                                const phi::KernelKey& kernel_key,
                                ir::Program* program) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  std::string op_name = paddle::dialect::PhiKernelOp::name();

  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

  if ((src_place.GetType() == phi::AllocationType::CPU) &&
      (dst_place.GetType() == phi::AllocationType::GPU)) {
    auto copy_kernel_key = kernel_key;
    copy_kernel_key.set_backend(phi::Backend::GPU);
    std::unordered_map<std::string, ir::Attribute> op_attribute{
        {"op_name", ir::StrAttribute::get(ctx, "pd.memcpy_h2d")},
        {"kernel_name", ir::StrAttribute::get(ctx, "memcpy_h2d")},
        {"kernel_key", dialect::KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", ir::Int32Attribute::get(ctx, 1)}};

    ir::Operation* op =
        ir::Operation::Create({in}, op_attribute, {out_type}, op_info);

    program->block()->push_back(op);

    auto new_in = op->result(0);

    return new_in;
  } else if ((src_place.GetType() == phi::AllocationType::GPU) &&
             (dst_place.GetType() == phi::AllocationType::CPU)) {
    auto copy_kernel_key = kernel_key;
    copy_kernel_key.set_backend(phi::Backend::GPU);
    std::unordered_map<std::string, ir::Attribute> op_attribute{
        {"op_name", ir::StrAttribute::get(ctx, "pd.memcpy_d2h")},
        {"kernel_name", ir::StrAttribute::get(ctx, "memcpy_d2h")},
        {"kernel_key", dialect::KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", ir::Int32Attribute::get(ctx, 0)}};

    ir::Operation* op =
        ir::Operation::Create({in}, op_attribute, {out_type}, op_info);

    program->block()->push_back(op);

    auto new_in = op->result(0);
    return new_in;
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support cpu to gpu and gpu to cpu"));
  }
}

phi::KernelKey GetKernelKey(
    ir::Operation* op,
    const phi::Place& place,
    const std::unordered_map<ir::Value, ir::OpResult>& map_value_pair,
    dialect::OpYamlInfoParser* op_info_parser = nullptr) {
  if (op->name() == "pd.feed") {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    return {phi::Backend::CPU,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->name() == "pd.data") {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    auto t =
        op->attributes().at("place").dyn_cast<dialect::PlaceAttribute>().data();

    auto backend = paddle::experimental::ParseBackend(t);

    return {backend,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;
  if (op_info_parser != nullptr) {
    // only suppurt non vector input for now
    int tensor_input_number = op_info_parser->InputTensorNumber();

    auto attr_map = op->attributes();
    auto& data_type_info = op_info_parser->OpRuntimeInfo().kernel_key_dtype;

    if (!data_type_info.empty() && !data_type_info[0].empty()) {
      // only support single input and attribute
      auto slot_name = data_type_info[0];
      auto& input_map = op_info_parser->InputName2Id();

      auto find_it = Str2PhiDataType.find(slot_name);
      if (find_it != Str2PhiDataType.end()) {
        kernel_data_type = find_it->second;
      } else if (input_map.count(slot_name)) {
        // parse from input
        int in_index = input_map.at(slot_name);
        auto type = map_value_pair.at(op->operand_source(in_index)).type();

        if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
          kernel_data_type = TransToPhiDataType(
              type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                  .dtype());
        } else if (type.isa<ir::VectorType>()) {
          auto vec_data = type.dyn_cast<ir::VectorType>().data();
          if (vec_data.empty()) {
            kernel_data_type = phi::DataType::UNDEFINED;
          } else {
            if (vec_data[0].isa<paddle::dialect::AllocatedDenseTensorType>()) {
              kernel_data_type = TransToPhiDataType(
                  vec_data[0]
                      .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                      .dtype());
            } else {
              PADDLE_THROW(phi::errors::Unimplemented(
                  "Only support DenseTensorType in vector"));
            }
          }
        } else if (type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
          kernel_data_type = TransToPhiDataType(
              type.dyn_cast<paddle::dialect::AllocatedSelectedRowsType>()
                  .dtype());
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Only support DenseTensorType, SelectedRows, VectorType"));
        }

      } else {
        PADDLE_ENFORCE_EQ(attr_map.count(slot_name),
                          true,
                          phi::errors::PreconditionNotMet(
                              "[%s] MUST in attribute map", slot_name));

        auto attr_type = op_info_parser->AttrTypeName(slot_name);
        PADDLE_ENFORCE_EQ(attr_type,
                          "paddle::dialect::DataTypeAttribute",
                          phi::errors::PreconditionNotMet(
                              "Type of [%s] should be DataType", slot_name));
        kernel_data_type = attr_map.at(slot_name)
                               .dyn_cast<paddle::dialect::DataTypeAttribute>()
                               .data();
      }
    }

    // parse all the input tensor
    if (tensor_input_number == 0 || op->name() == "pd.full_") {
      // all the information have to get from attribute and context

      if (op->name() == "pd.uniform") {
        // try to process uniform, use shape to determin backend
        // TODO(phlrain): shuold support other initilize op
        auto define_op = op->operand_source(0).GetDefiningOp();
        if (define_op->name() == "pd.full_int_array") {
          auto shape = define_op->attributes()
                           .at("value")
                           .dyn_cast<dialect::IntArrayAttribute>()
                           .data()
                           .GetData();

          size_t numel = 1;
          for (auto& s : shape) {
            numel *= s;
          }
          if (numel > init_on_gpu_threashold) {
            kernel_backend = phi::Backend::GPU;
          }
        }
      }

      if (kernel_backend == phi::Backend::UNDEFINED) {
        kernel_backend = paddle::experimental::ParseBackend(place);
      }
    }
  }

  if (op->num_operands() > 0) {
    paddle::experimental::detail::KernelKeyParser kernel_key_parser;

    for (size_t i = 0; i < op->num_operands(); ++i) {
      // NOTE, only op with OpYamlInfo can have TensorArr
      if (op_info_parser != nullptr && op_info_parser->IsTensorAttribute(i)) {
        continue;
      }
      auto input_tmp = op->operand_source(i);
      // NOTE: if not input_tmp, it's an optional input
      if (!input_tmp) {
        continue;
      }
      auto new_input_tmp = map_value_pair.at(input_tmp);

      auto input_type = new_input_tmp.type();
      dialect::AllocatedDenseTensorType type;
      if (input_type.isa<dialect::AllocatedDenseTensorType>()) {
        type = input_type.dyn_cast<dialect::AllocatedDenseTensorType>();
      } else if (input_type.isa<ir::VectorType>()) {
        type = input_type.dyn_cast<ir::VectorType>()[0]
                   .dyn_cast<dialect::AllocatedDenseTensorType>();
      }

      // fake tensor here
      auto ptr = new phi::Allocation(nullptr, 0, type.place());

      std::shared_ptr<phi::Allocation> holder(ptr);

      auto dtype = TransToPhiDataType(type.dtype());

      phi::DenseTensorMeta meta(
          dtype, type.dims(), type.data_layout(), type.lod(), type.offset());

      phi::DenseTensor fake_tensor(holder, meta);

      kernel_key_parser.AssignKernelKeySet(fake_tensor);
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
  }

  if (kernel_backend == phi::Backend::UNDEFINED) {
    kernel_backend = paddle::experimental::ParseBackend(place);
  }

  phi::KernelKey res(kernel_backend, kernel_layout, kernel_data_type);
  return res;
}

std::unique_ptr<ir::Program> PdOpLowerToKernelPass(ir::Program* prog,
                                                   phi::Place place) {
  auto program = std::make_unique<ir::Program>(ir::IrContext::Instance());

  auto block = prog->block();

  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleKernelDialect>();

  std::unordered_map<ir::Operation*, ir::Operation*> map_op_pair;
  std::unordered_map<ir::Value, ir::OpResult> map_value_pair;

  std::string op_name = paddle::dialect::PhiKernelOp::name();

  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

  for (auto op_item : *block) {
    VLOG(6) << "op name " << op_item->name();
    paddle::dialect::OpYamlInfoInterface op_info_interface =
        op_item->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
    std::unique_ptr<OpYamlInfoParser> op_info_parser(nullptr);
    if (op_info_interface) {
      op_info_parser =
          std::make_unique<OpYamlInfoParser>(op_info_interface.GetOpInfo());
    }

    std::string kernel_fn_str;
    if (op_info_parser != nullptr) {
      kernel_fn_str = op_info_parser->OpRuntimeInfo().kernel_func[0];
    }

    auto kernel_key =
        GetKernelKey(op_item, place, map_value_pair, op_info_parser.get());
    VLOG(6) << "kernel type " << kernel_key;

    if (NeedFallBackCpu((op_item), kernel_fn_str, kernel_key)) {
      kernel_key.set_backend(phi::Backend::CPU);
    }

    if (NeedFallBackFromGPUDNN2GPU(op_item, kernel_key)) {
      kernel_key.set_backend(phi::Backend::GPU);
    }

    // only for single output
    // need update new kernel key layout and data tyep

    std::vector<ir::Type> op_output_types;

    if (op_item->num_results() > 0) {
      auto phi_kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
          kernel_fn_str, kernel_key);
      auto args_def = phi_kernel.args_def();
      auto output_defs = args_def.output_defs();
      if (!UnchangeOutputOps.count(op_item->name()) &&
          !LegacyOpList.count(op_item->name())) {
        PADDLE_ENFORCE_EQ(
            op_item->num_results(),
            output_defs.size(),
            phi::errors::PreconditionNotMet(
                "op [%s] kernel output args defs should equal op outputs",
                op_item->name()));
      }

      for (size_t i = 0; i < op_item->num_results(); ++i) {
        phi::Place out_place;
        if ((!UnchangeOutputOps.count(op_item->name())) &&
            (!LegacyOpList.count(op_item->name())) && phi_kernel.IsValid()) {
          out_place = phi::TransToPhiPlace(output_defs[i].backend);
        } else {
          out_place = phi::TransToPhiPlace(kernel_key.backend());
        }

        auto result_type = op_item->result(i).type();
        if (!result_type) {
          op_output_types.push_back(result_type);
        } else if (result_type.isa<dialect::DenseTensorType>()) {
          auto allocated_dense_tensor_dtype =
              paddle::dialect::AllocatedDenseTensorType::get(
                  ctx,
                  out_place,
                  result_type.dyn_cast<dialect::DenseTensorType>());
          op_output_types.push_back(allocated_dense_tensor_dtype);
        } else if (result_type.isa<ir::VectorType>()) {
          std::vector<ir::Type> vec_inner_types;
          auto base_types = result_type.dyn_cast<ir::VectorType>().data();
          for (auto& base_type : base_types) {
            if (base_type) {
              if (base_type.isa<dialect::DenseTensorType>()) {
                auto allocated_dense_tensor_dtype =
                    paddle::dialect::AllocatedDenseTensorType::get(
                        ctx,
                        out_place,
                        base_type.dyn_cast<dialect::DenseTensorType>());
                vec_inner_types.push_back(allocated_dense_tensor_dtype);
              } else {
                PADDLE_THROW(phi::errors::Unimplemented(
                    "only support dense tensor in vector type for now"));
              }
            } else {
              // NOTE(phlrain), kernel not support a nullptr in output
              ir::Type fp32_dtype = ir::Float32Type::get(ctx);
              phi::DDim dims = {};
              phi::DataLayout data_layout = phi::DataLayout::NCHW;
              phi::LoD lod = {{}};
              size_t offset = 0;
              auto dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
                  ctx, fp32_dtype, dims, data_layout, lod, offset);
              auto allocated_dense_tensor_dtype =
                  paddle::dialect::AllocatedDenseTensorType::get(
                      ctx, out_place, dense_tensor_dtype);
              vec_inner_types.push_back(allocated_dense_tensor_dtype);
            }
          }

          ir::Type t1 = ir::VectorType::get(ctx, vec_inner_types);
          op_output_types.push_back(t1);
        } else if (result_type.isa<dialect::SelectedRowsType>()) {
          auto allocated_selected_rows_dtype =
              paddle::dialect::AllocatedSelectedRowsType::get(
                  ctx,
                  out_place,
                  result_type.dyn_cast<dialect::SelectedRowsType>());
          op_output_types.emplace_back(allocated_selected_rows_dtype);
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Result type only support DenseTensorType and VectorType"));
        }
      }
    }

    // constuct input
    std::vector<ir::OpResult> vec_inputs;

    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        PADDLE_ENFORCE_EQ(map_value_pair.count(cur_in),
                          true,
                          phi::errors::PreconditionNotMet(
                              "[%d]'s input of [%s] op MUST in map pair",
                              i,
                              op_item->name()));
        auto new_in = map_value_pair.at(cur_in);

        auto new_in_type = new_in.type();

        auto& kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
            kernel_fn_str, kernel_key);

        if (kernel.IsValid() && (!UnchangeOutputOps.count(op_item->name()))) {
          if (new_in_type.isa<dialect::AllocatedDenseTensorType>()) {
            // allocated type
            auto place =
                new_in_type.dyn_cast<dialect::AllocatedDenseTensorType>()
                    .place();

            // get input args def type
            auto args_def = kernel.args_def();
            auto input_defs = args_def.input_defs();

            bool need_trans =
                (place.GetType() != phi::AllocationType::UNDEFINED) &&
                (op_info_parser != nullptr &&
                 !op_info_parser->IsTensorAttribute(i)) &&
                (paddle::experimental::NeedTransformPlace(
                    place, kernel.InputAt(i).backend, {}));
            if (need_trans) {
              VLOG(6) << "need trans from " << place << " to "
                      << kernel_key.backend();
              // build memcopy op
              new_in = AddPlaceTransferOp(
                  new_in,
                  new_in_type,
                  place,
                  phi::TransToPhiPlace(kernel.InputAt(i).backend),
                  kernel_key,
                  program.get());
            }
          } else if (new_in_type.isa<ir::VectorType>()) {
            // [ todo need update here, support combine data transfomer]
          } else if (new_in_type.isa<dialect::AllocatedSelectedRowsType>()) {
            // do nothing here
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "only support allocated dense tensor type for now"));
          }
        }
        vec_inputs.push_back(new_in);
      }
    }

    std::unordered_map<std::string, ir::Attribute> op_attribute{
        {"op_name", ir::StrAttribute::get(ctx, op_item->name())},
        {"kernel_name", ir::StrAttribute::get(ctx, kernel_fn_str)},
        {"kernel_key", dialect::KernelAttribute::get(ctx, kernel_key)}};

    auto op_attr_map = op_item->attributes();

    for (auto& map_item : op_attr_map) {
      op_attribute.emplace(map_item.first, map_item.second);
    }

    if (op_item->HasTrait<paddle::dialect::InplaceTrait>()) {
      op_attribute.emplace("is_inplace", ir::BoolAttribute::get(ctx, true));
    }

    ir::Operation* op = ir::Operation::Create(
        vec_inputs, op_attribute, op_output_types, op_info);

    map_op_pair[op_item] = op;

    // only deal with single output
    if (op_item->num_results() > 0) {
      for (size_t i = 0; i < op_item->num_results(); ++i) {
        map_value_pair[op_item->result(i)] = op->result(i);
      }
    }

    program->block()->push_back(op);

    if (op_item->name() == "pd.feed" && platform::is_gpu_place(place)) {
      // add shadow feed op
      phi::KernelKey shadow_key{
          phi::Backend::GPU,
          phi::DataLayout::ANY,
          TransToPhiDataType(
              op_item->result(0).type().dyn_cast<DenseTensorType>().dtype())};
      std::unordered_map<std::string, ir::Attribute> attr_map{
          {"op_name", ir::StrAttribute::get(ctx, "pd.shadow_feed")},
          {"kernel_name", ir::StrAttribute::get(ctx, "shadow_feed")},
          {"kernel_key", dialect::KernelAttribute::get(ctx, shadow_key)}};

      auto out_type = paddle::dialect::AllocatedDenseTensorType::get(
          ctx,
          phi::TransToPhiPlace(shadow_key.backend()),
          op_item->result(0).type().dyn_cast<dialect::DenseTensorType>());

      ir::Operation* shadow_op =
          ir::Operation::Create({op->result(0)}, attr_map, {out_type}, op_info);

      map_op_pair[op_item] = shadow_op;
      program->block()->push_back(shadow_op);
      if (op_item->num_results() > 0) {
        for (size_t i = 0; i < shadow_op->num_results(); ++i) {
          map_value_pair[op_item->result(i)] = shadow_op->result(i);
        }
      }
    }
  }

  return program;
}

}  // namespace dialect
}  // namespace paddle
