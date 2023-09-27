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

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/builtin_op.h"

namespace paddle {
namespace dialect {

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
    "pd_op.data",
    "builtin.combine",
    "builtin.slice",
    "builtin.split",
    "pd_op.feed",
    "pd_op.fetch",
    "builtin.set_parameter",
    "builtin.get_parameter",
    "pd_op.shadow_output"};

const std::unordered_set<std::string> SpecialLowerOps = {"builtin.combine",
                                                         "builtin.slice",
                                                         "builtin.split",
                                                         "pd_op.if",
                                                         "cf.yield"};

bool NeedFallBackCpu(const pir::Operation* op,
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

phi::Backend GetDstBackend(const std::string& op_name,
                           phi::Place place,
                           const OpYamlInfoParser* op_yaml_info_parser,
                           phi::Backend kernel_def_backend,
                           size_t input_index) {
  if (op_name == "builtin.set_parameter" &&
      place.GetType() == phi::AllocationType::GPU) {
    // NOTE: align old executor, all the paramter are initilizered
    // on backend of executor place defined
    return phi::TransToPhiBackend(place);
  }

  auto dst_backend = kernel_def_backend;
  if (op_yaml_info_parser != nullptr &&
      op_yaml_info_parser->IsTensorAttribute(input_index)) {
    // Tensor Attribute should on cpu backend for better performance
    dst_backend = phi::Backend::CPU;
  }

  return dst_backend;
}

bool NeedFallBackFromGPUDNN2GPU(pir::Operation* op,
                                const phi::KernelKey kernel_key) {
  // NOTE(phlrain): keep the same kernel select strategy with
  // GetExepectKernelKey
  if (op->isa<paddle::dialect::Pool2dOp>() ||
      op->isa<paddle::dialect::Pool2dGradOp>()) {
    if (kernel_key.backend() == phi::Backend::GPUDNN &&
        (op->attributes()
             .at("adaptive")
             .dyn_cast<pir::BoolAttribute>()
             .data() == true)) {
      return true;
    }
  }

  return false;
}

std::set<std::string> GetSkipFeedNames(pir::Block* block) {
  std::set<std::string> data_op_names;
  for (auto op_item : *block) {
    if (op_item->isa<paddle::dialect::DataOp>()) {
      data_op_names.insert(op_item->attributes()
                               .at("name")
                               .dyn_cast<pir::StrAttribute>()
                               .AsString());
    }
  }
  return data_op_names;
}

bool SkipFeedOp(pir::Operation* op, const std::set<std::string>& feed_names) {
  return feed_names.count(
      op->attributes().at("name").dyn_cast<pir::StrAttribute>().AsString());
}

std::vector<std::shared_ptr<phi::TensorBase>> GetFakeTensorList(
    pir::Value new_input_tmp) {
  std::vector<std::shared_ptr<phi::TensorBase>> vec_res;
  auto input_type = new_input_tmp.type();

  auto build_fake_dense_tensor =
      [](const dialect::AllocatedDenseTensorType& type) {
        auto ptr = new phi::Allocation(nullptr, 0, type.place());

        std::shared_ptr<phi::Allocation> holder(ptr);

        auto dtype = TransToPhiDataType(type.dtype());

        phi::DenseTensorMeta meta(
            dtype, type.dims(), type.data_layout(), type.lod(), type.offset());

        return std::make_shared<phi::DenseTensor>(holder, meta);
      };

  auto build_fake_selected_rows =
      [](const dialect::AllocatedSelectedRowsType& type) {
        auto ptr = new phi::Allocation(nullptr, 0, type.place());

        std::shared_ptr<phi::Allocation> holder(ptr);

        auto dtype = TransToPhiDataType(type.dtype());

        phi::DenseTensorMeta meta(
            dtype, type.dims(), type.data_layout(), type.lod(), type.offset());

        std::vector<int64_t> rows;
        int64_t height = 0;
        rows.clear();

        auto sr = std::make_shared<phi::SelectedRows>(rows, height);

        phi::DenseTensor dense_tensor(holder, meta);
        *(sr->mutable_value()) = dense_tensor;

        return sr;
      };

  if (input_type.isa<dialect::AllocatedDenseTensorType>()) {
    vec_res.push_back(build_fake_dense_tensor(
        input_type.dyn_cast<dialect::AllocatedDenseTensorType>()));
  } else if (input_type.isa<dialect::AllocatedSelectedRowsType>()) {
    vec_res.push_back(build_fake_selected_rows(
        input_type.dyn_cast<dialect::AllocatedSelectedRowsType>()));
  } else if (input_type.isa<pir::VectorType>()) {
    auto vec_inner_types = input_type.dyn_cast<pir::VectorType>().data();
    for (size_t i = 0; i < vec_inner_types.size(); ++i) {
      if (vec_inner_types[i].isa<dialect::AllocatedDenseTensorType>()) {
        vec_res.push_back(build_fake_dense_tensor(
            vec_inner_types[i].dyn_cast<dialect::AllocatedDenseTensorType>()));
      } else if (vec_inner_types[i].isa<dialect::AllocatedSelectedRowsType>()) {
        vec_res.push_back(build_fake_selected_rows(
            vec_inner_types[i].dyn_cast<dialect::AllocatedSelectedRowsType>()));
      }
    }
  }

  return vec_res;
}

pir::OpResult AddPlaceTransferOp(pir::OpResult in,
                                 pir::Type out_type,
                                 const phi::Place& src_place,
                                 const phi::Place& dst_place,
                                 const phi::KernelKey& kernel_key,
                                 pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = paddle::dialect::PhiKernelOp::name();

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

  if ((src_place.GetType() == phi::AllocationType::CPU) &&
      (dst_place.GetType() == phi::AllocationType::GPU)) {
    auto copy_kernel_key = kernel_key;
    copy_kernel_key.set_backend(phi::Backend::GPU);
    std::unordered_map<std::string, pir::Attribute> op_attribute{
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.memcpy_h2d")},
        {"kernel_name", pir::StrAttribute::get(ctx, "memcpy_h2d")},
        {"kernel_key", dialect::KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};

    pir::Operation* op =
        pir::Operation::Create({in}, op_attribute, {out_type}, op_info);

    if (in.owner()->HasAttribute(kAttrIsPersisable)) {
      op->set_attribute(kAttrIsPersisable,
                        in.owner()->attribute(kAttrIsPersisable));
    }
    block->push_back(op);

    auto new_in = op->result(0);

    return new_in;
  } else if ((src_place.GetType() == phi::AllocationType::GPU) &&
             (dst_place.GetType() == phi::AllocationType::CPU)) {
    auto copy_kernel_key = kernel_key;
    copy_kernel_key.set_backend(phi::Backend::GPU);
    std::unordered_map<std::string, pir::Attribute> op_attribute{
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.memcpy_d2h")},
        {"kernel_name", pir::StrAttribute::get(ctx, "memcpy_d2h")},
        {"kernel_key", dialect::KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 0)}};

    pir::Operation* op =
        pir::Operation::Create({in}, op_attribute, {out_type}, op_info);

    block->push_back(op);

    auto new_in = op->result(0);
    return new_in;
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support cpu to gpu and gpu to cpu"));
  }
}

pir::Type BuildOutputType(pir::Type type,
                          const phi::Place& place,
                          phi::DataType data_type,
                          pir::IrContext* ctx) {
  if (type.isa<dialect::DenseTensorType>()) {
    auto dense_tensor_type = type.dyn_cast<dialect::DenseTensorType>();
    auto out_dtype = dense_tensor_type.dtype();

    // TODO(phlrain): open this after fix pr(55509) confict
    // if (data_type != phi::DataType::UNDEFINED) {
    //   out_dtype = TransToIrDataType(data_type, ctx);
    // }

    return dialect::AllocatedDenseTensorType::get(
        ctx,
        place,
        out_dtype,
        dense_tensor_type.dims(),
        dense_tensor_type.data_layout(),
        dense_tensor_type.lod(),
        dense_tensor_type.offset());

  } else if (type.isa<dialect::SelectedRowsType>()) {
    auto selected_rows_type = type.dyn_cast<dialect::SelectedRowsType>();
    auto out_dtype = selected_rows_type.dtype();

    // TODO(phlrain): open this after fix pr(55509) confict
    // if (data_type != phi::DataType::UNDEFINED) {
    //   out_dtype = TransToIrDataType(data_type, ctx);
    // }
    return dialect::AllocatedSelectedRowsType::get(
        ctx,
        place,
        out_dtype,
        selected_rows_type.dims(),
        selected_rows_type.data_layout(),
        selected_rows_type.lod(),
        selected_rows_type.offset());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType and SelectedRowsType"));
  }
}

phi::DataType GetKernelDataTypeByYamlInfo(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::OpResult>& map_value_pair,
    const dialect::OpYamlInfoParser* op_info_parser) {
  auto& attr_map = op->attributes();
  auto& data_type_info = op_info_parser->OpRuntimeInfo().kernel_key_dtype;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;

  for (size_t i = 0; i < data_type_info.size(); ++i) {
    auto slot_name = data_type_info[i];
    auto& input_map = op_info_parser->InputName2Id();

    auto find_it = Str2PhiDataType.find(slot_name);
    if (find_it != Str2PhiDataType.end()) {
      kernel_data_type = find_it->second;
    } else if (input_map.count(slot_name)) {
      // parse from input
      int in_index = static_cast<int>(input_map.at(slot_name));
      auto type = map_value_pair.at(op->operand_source(in_index)).type();

      if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>().dtype());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_data_type = phi::DataType::UNDEFINED;
        } else {
          if (vec_data[0].isa<paddle::dialect::AllocatedDenseTensorType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0]
                    .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                    .dtype());
          } else if (vec_data[0]
                         .isa<paddle::dialect::AllocatedSelectedRowsType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0]
                    .dyn_cast<paddle::dialect::AllocatedSelectedRowsType>()
                    .dtype());
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "Only support DenseTensorType and SelectedRowsType in vector"));
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

    if (kernel_data_type != phi::DataType::UNDEFINED) {
      // In yaml definition, data type have an order
      // like: data_type : dtype > x
      // Should break when found a defined data type
      break;
    }
  }

  return kernel_data_type;
}

phi::Backend GetKernelBackendByYamlInfo(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::OpResult>& map_value_pair,
    const dialect::OpYamlInfoParser* op_info_parser,
    const phi::Place& place) {
  auto& attr_map = op->attributes();
  auto& backend_info = op_info_parser->OpRuntimeInfo().kernel_key_backend;
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  for (size_t i = 0; i < backend_info.size(); ++i) {
    auto slot_name = backend_info[i];
    auto& input_map = op_info_parser->InputName2Id();

    if (input_map.count(slot_name)) {
      // parse from input
      int in_index = static_cast<int>(input_map.at(slot_name));
      auto type = map_value_pair.at(op->operand_source(in_index)).type();

      if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>().place());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_backend = phi::Backend::UNDEFINED;
        } else {
          if (vec_data[0].isa<paddle::dialect::AllocatedDenseTensorType>()) {
            kernel_backend = paddle::experimental::ParseBackend(
                vec_data[0]
                    .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                    .place());
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "Only support DenseTensorType in vector"));
          }
        }
      } else if (type.isa<paddle::dialect::AllocatedSelectedRowsType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<paddle::dialect::AllocatedSelectedRowsType>()
                .place());
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
                        "paddle::dialect::PlaceAttribute",
                        phi::errors::PreconditionNotMet(
                            "Type of [%s] should be DataType", slot_name));
      kernel_backend = paddle::experimental::ParseBackend(
          attr_map.at(slot_name)
              .dyn_cast<paddle::dialect::PlaceAttribute>()
              .data());
    }
    if (kernel_backend != phi::Backend::UNDEFINED) {
      // In yaml definition, backend have an order
      // like: backend : place > x
      // Should break when found a defined data type
      break;
    }
  }

  if (backend_info.size() > 0 && kernel_backend == phi::Backend::UNDEFINED) {
    kernel_backend = paddle::experimental::ParseBackend(place);
  }

  return kernel_backend;
}

phi::KernelKey GetKernelKey(
    pir::Operation* op,
    const phi::Place& place,
    const std::string& kernel_fn_str,
    const std::unordered_map<pir::Value, pir::OpResult>& map_value_pair,
    dialect::OpYamlInfoParser* op_info_parser = nullptr) {
  if (op->isa<paddle::dialect::FeedOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    return {phi::Backend::CPU,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->isa<paddle::dialect::DataOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    auto data_place =
        op->attributes().at("place").dyn_cast<dialect::PlaceAttribute>().data();

    auto backend = paddle::experimental::ParseBackend(data_place);

    return {backend,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->name() == "pd_op.seed") {
    auto backend = paddle::experimental::ParseBackend(place);
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
    int tensor_input_number =
        static_cast<int>(op_info_parser->InputTensorNumber());

    // get datatype info
    kernel_data_type =
        GetKernelDataTypeByYamlInfo(op, map_value_pair, op_info_parser);
    kernel_backend =
        GetKernelBackendByYamlInfo(op, map_value_pair, op_info_parser, place);

    // parse all the input tensor
    if (tensor_input_number == 0 || op->isa<paddle::dialect::Full_Op>()) {
      // all the information have to get from attribute and context

      if (op->isa<paddle::dialect::UniformOp>()) {
        // try to process uniform, use shape to determin backend
        // TODO(phlrain): shuold support other initilize op
        auto define_op =
            op->operand_source(0).dyn_cast<pir::OpResult>().owner();
        if (define_op->isa<paddle::dialect::FullIntArrayOp>()) {
          auto shape = define_op->attribute<dialect::IntArrayAttribute>("value")
                           .data()
                           .GetData();
        }
      }

      if (kernel_backend == phi::Backend::UNDEFINED) {
        kernel_backend = paddle::experimental::ParseBackend(place);
      }
    }
  }

  if ((kernel_backend == phi::Backend::UNDEFINED ||
       kernel_data_type == phi::DataType::UNDEFINED) &&
      op->num_operands() > 0) {
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

      auto fake_tensors = GetFakeTensorList(new_input_tmp);
      for (auto& fake_tensor : fake_tensors) {
        kernel_key_parser.AssignKernelKeySet(*fake_tensor);
      }

      // Because we can't make sure the place when build data op
      // and the output place of data op is undefined. It means we
      // don't know how to select the kernel in the next of op that
      // uses data op outout as inputs. So, we need set kernel backend
      // manually.
      if (op->operand_source(i)
              .dyn_cast<pir::OpResult>()
              .owner()
              ->isa<paddle::dialect::DataOp>()) {
        auto data_op = op->operand_source(i).dyn_cast<pir::OpResult>().owner();
        auto data_place =
            data_op->attribute<dialect::PlaceAttribute>("place").data();

        auto data_op_backend = paddle::experimental::ParseBackend(data_place);
        if (data_op_backend == phi::Backend::UNDEFINED) {
          data_op_backend = paddle::experimental::ParseBackend(place);
        }
        kernel_key_parser.key_set.backend_set =
            kernel_key_parser.key_set.backend_set |
            paddle::experimental::BackendSet(data_op_backend);
      } else if (op->operand_source(i)
                     .dyn_cast<pir::OpResult>()
                     .owner()
                     ->isa<pir::CombineOp>()) {
        auto combine_op =
            op->operand_source(i).dyn_cast<pir::OpResult>().owner();
        for (size_t j = 0; j < combine_op->num_operands(); ++j) {
          if (combine_op->operand_source(j)
                  .dyn_cast<pir::OpResult>()
                  .owner()
                  ->isa<DataOp>()) {
            auto data_op =
                combine_op->operand_source(j).dyn_cast<pir::OpResult>().owner();
            auto data_place =
                data_op->attribute<PlaceAttribute>("place").data();

            auto data_op_backend =
                paddle::experimental::ParseBackend(data_place);
            if (data_op_backend == phi::Backend::UNDEFINED) {
              data_op_backend = paddle::experimental::ParseBackend(place);
            }
            kernel_key_parser.key_set.backend_set =
                kernel_key_parser.key_set.backend_set |
                paddle::experimental::BackendSet(data_op_backend);
            break;
          }
        }
      }
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

  if (op->isa<paddle::dialect::LoadCombineOp>()) {
    res.set_dtype(phi::DataType::FLOAT32);
  }
  if (NeedFallBackCpu((op), kernel_fn_str, res)) {
    res.set_backend(phi::Backend::CPU);
  }

  if (NeedFallBackFromGPUDNN2GPU(op, res)) {
    res.set_backend(phi::Backend::GPU);
  }

  return res;
}

void HandleForIfOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair) {
  auto cur_in = op_item->operand_source(0);

  PADDLE_ENFORCE_EQ(
      map_value_pair->count(cur_in),
      true,
      phi::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
  auto new_in = map_value_pair->at(cur_in);

  pir::Builder builder(ctx, block);

  auto base_if_op = op_item->dyn_cast<paddle::dialect::IfOp>();
  std::vector<pir::Type> op_output_types;
  for (size_t i = 0; i < base_if_op.num_results(); ++i) {
    op_output_types.push_back(paddle::dialect::AllocatedDenseTensorType::get(
        ctx,
        place,
        base_if_op.result(i).type().dyn_cast<dialect::DenseTensorType>()));
  }
  auto new_if_op =
      builder.Build<paddle::dialect::IfOp>(new_in, std::move(op_output_types));

  // process true block
  pir::Block* true_block = new_if_op.true_block();
  ProcessBlock(place,
               base_if_op.true_block(),
               true_block,
               ctx,
               map_op_pair,
               map_value_pair);

  // process false block
  pir::Block* false_block = new_if_op.false_block();
  ProcessBlock(place,
               base_if_op.false_block(),
               false_block,
               ctx,
               map_op_pair,
               map_value_pair);
}

pir::OpResult GetNewInput(
    const pir::Value cur_in,
    const std::unordered_map<pir::Value, pir::OpResult>& map_value_pair,
    const int index,
    const std::string op_name) {
  PADDLE_ENFORCE_EQ(
      map_value_pair.count(cur_in),
      true,
      phi::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST be in map pair", index, op_name));
  auto new_in = map_value_pair.at(cur_in);
  return new_in;
}

void HandleForSpecialOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair) {
  if (op_item->isa<paddle::dialect::IfOp>()) {
    HandleForIfOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
    return;
  }
  std::vector<pir::Value> vec_inputs;
  std::vector<pir::Type> op_output_types;
  if (op_item->isa<::pir::CombineOp>()) {
    // Copy op inputs
    std::vector<pir::Type> vec_inner_types;
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(cur_in, *map_value_pair, i, op_item->name());
        vec_inputs.push_back(new_in);
        vec_inner_types.push_back(new_in.type());
      }
    }
    // Copy op output type

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types.push_back(t1);
  }

  if (op_item->isa<::pir::SliceOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(cur_in, *map_value_pair, i, op_item->name());
        vec_inputs.push_back(new_in);

        if (new_in.type().isa<pir::VectorType>()) {
          auto vec_types = new_in.type().dyn_cast<pir::VectorType>().data();
          auto index = op_item->attribute("index")
                           .dyn_cast<pir::Int32Attribute>()
                           .data();
          op_output_types.push_back(vec_types[index]);
        } else {
          PADDLE_THROW(
              phi::errors::Unimplemented("only support vector type for now"));
        }
      }
    }
  }

  if (op_item->isa<::pir::SplitOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(cur_in, *map_value_pair, i, op_item->name());
        vec_inputs.push_back(new_in);

        if (new_in.type().isa<pir::VectorType>()) {
          auto vec_types = new_in.type().dyn_cast<pir::VectorType>().data();
          for (uint64_t idx = 0; idx < vec_types.size(); idx++) {
            op_output_types.push_back(vec_types[idx]);
          }
        } else {
          PADDLE_THROW(
              phi::errors::Unimplemented("only support vector type for now"));
        }
      }
    }
  }

  if (op_item->name() == "cf.yield") {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(cur_in, *map_value_pair, i, op_item->name());
        vec_inputs.push_back(new_in);
      }
    }
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_item->name());
  // Generate new op
  pir::Operation* op = pir::Operation::Create(
      vec_inputs, op_item->attributes(), op_output_types, op_info);
  block->push_back(op);
  (*map_op_pair)[op_item] = op;
  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  VLOG(6) << "Deep copy a new builtin op: " << op_item->name();
}

std::vector<pir::Type> BuildOpOutputType(pir::Operation* op_item,
                                         const std::string& kernel_fn_str,
                                         const phi::KernelKey& kernel_key,
                                         pir::IrContext* ctx) {
  if (op_item->num_results() == 0) {
    return {};
  }
  std::vector<pir::Type> op_output_types;
  auto phi_kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
      kernel_fn_str, kernel_key);
  auto args_def = phi_kernel.args_def();
  auto output_defs = args_def.output_defs();
  if (!UnchangeOutputOps.count(op_item->name()) &&
      !IsLegacyOp(op_item->name())) {
    PADDLE_ENFORCE_EQ(
        op_item->num_results(),
        output_defs.size(),
        phi::errors::PreconditionNotMet(
            "op [%s] kernel output args defs should equal op outputs",
            op_item->name()));
  }

  for (size_t i = 0; i < op_item->num_results(); ++i) {
    phi::Place out_place = phi::TransToPhiPlace(kernel_key.backend());

    phi::DataType out_phi_dtype = phi::DataType::UNDEFINED;
    if ((!UnchangeOutputOps.count(op_item->name())) &&
        (!IsLegacyOp(op_item->name())) && phi_kernel.IsValid()) {
      out_place = phi::TransToPhiPlace(output_defs[i].backend);
      out_phi_dtype = output_defs[i].dtype;
    }

    auto result_type = op_item->result(i).type();
    if (!result_type) {
      op_output_types.push_back(result_type);
    } else if (result_type.isa<dialect::DenseTensorType>() ||
               result_type.isa<dialect::SelectedRowsType>()) {
      op_output_types.push_back(
          BuildOutputType(result_type, out_place, out_phi_dtype, ctx));
    } else if (result_type.isa<pir::VectorType>()) {
      std::vector<pir::Type> vec_inner_types;
      auto base_types = result_type.dyn_cast<pir::VectorType>().data();
      for (auto& base_type : base_types) {
        if (base_type) {
          if (base_type.isa<dialect::DenseTensorType>() ||
              base_type.isa<dialect::SelectedRowsType>()) {
            vec_inner_types.push_back(
                BuildOutputType(base_type, out_place, out_phi_dtype, ctx));
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "only support dense tensor and selected rows in vector type "
                "for now"));
          }
        } else {
          // NOTE(phlrain), kernel not support a nullptr in output
          pir::Type fp32_dtype = pir::Float32Type::get(ctx);
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

      pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
      op_output_types.push_back(t1);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Result type only support DenseTensorType, SelectedRowType and "
          "VectorType"));
    }
  }

  return op_output_types;
}

std::vector<pir::Value> BuildOpInputList(
    pir::Operation* op_item,
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const phi::Place place,
    const OpYamlInfoParser* op_info_parser,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair,
    pir::Block* block) {
  if (op_item->num_operands() == 0) {
    return {};
  }

  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    if (!cur_in) {
      vec_inputs.emplace_back();
      continue;
    }
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        phi::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));
    auto new_in = map_value_pair->at(cur_in);

    auto new_in_type = new_in.type();

    auto& kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
        kernel_fn_str, kernel_key);

    bool check_place_transfer =
        (op_item->isa<::pir::SetParameterOp>()) ||
        (kernel.IsValid() && (!UnchangeOutputOps.count(op_item->name())));

    if (check_place_transfer) {
      if (new_in_type.isa<dialect::AllocatedDenseTensorType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<dialect::AllocatedDenseTensorType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto dst_backend = GetDstBackend(op_item->name(),
                                         place,
                                         op_info_parser,
                                         kernel.InputAt(i).backend,
                                         i);

        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<dialect::AllocatedDenseTensorType>();
          auto out_type = dialect::AllocatedDenseTensorType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.dims(),
              new_in_alloc_type.data_layout(),
              new_in_alloc_type.lod(),
              new_in_alloc_type.offset());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<pir::VectorType>()) {
        // [ todo need update here, support combine data transfomer]
        // deal with pre combine op
        auto pre_define_op = cur_in.dyn_cast<pir::OpResult>().owner();

        if (pre_define_op->isa<::pir::CombineOp>()) {
          std::vector<pir::Value> inner_inputs;
          std::vector<pir::Type> types_in_vec;
          bool is_trans = false;
          for (size_t j = 0; j < pre_define_op->num_operands(); ++j) {
            auto in_i = map_value_pair->at(pre_define_op->operand_source(j));
            auto in_i_type = in_i.type();
            phi::Place place;
            if (in_i_type.isa<dialect::AllocatedDenseTensorType>()) {
              place = in_i_type.dyn_cast<dialect::AllocatedDenseTensorType>()
                          .place();
            } else if (in_i_type.isa<dialect::AllocatedSelectedRowsType>()) {
              place = in_i_type.dyn_cast<dialect::AllocatedSelectedRowsType>()
                          .place();
            } else {
              PADDLE_THROW(phi::errors::Unimplemented(
                  "builtin.combine Input type only support "
                  "VectorType<DenseTensorType> and "
                  "VectorType<SelectedRowsType>"));
            }

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
              auto out_place = phi::TransToPhiPlace(kernel.InputAt(i).backend);
              pir::Type out_type;
              if (in_i_type.isa<dialect::AllocatedDenseTensorType>()) {
                out_type = dialect::AllocatedDenseTensorType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<dialect::DenseTensorType>());
              } else if (in_i_type.isa<dialect::AllocatedSelectedRowsType>()) {
                out_type = dialect::AllocatedSelectedRowsType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<dialect::SelectedRowsType>());
              } else {
                PADDLE_THROW(phi::errors::Unimplemented(
                    "builtin.combine Input type only support "
                    "VectorType<DenseTensorType> and "
                    "VectorType<SelectedRowsType>"));
              }
              in_i = AddPlaceTransferOp(
                  in_i, out_type, place, out_place, kernel_key, block);

              is_trans = true;
            }

            inner_inputs.push_back(in_i);
            types_in_vec.push_back(in_i.type());
          }
          if (is_trans) {
            // Add combine op
            std::string combine_op_name(pir::CombineOp::name());
            pir::OpInfo op_info = ctx->GetRegisteredOpInfo(combine_op_name);

            pir::Type target_vec_type = pir::VectorType::get(ctx, types_in_vec);
            pir::Operation* operation = pir::Operation::Create(
                inner_inputs, {}, {target_vec_type}, op_info);

            new_in = operation->result(0);
            block->push_back(operation);
          }
        }
      } else if (new_in_type.isa<dialect::AllocatedSelectedRowsType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<dialect::AllocatedSelectedRowsType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto dst_backend = GetDstBackend(op_item->name(),
                                         place,
                                         op_info_parser,
                                         kernel.InputAt(i).backend,
                                         i);

        bool need_trans =
            (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
            (paddle::experimental::NeedTransformPlace(
                in_place, dst_backend, {}));
        if (need_trans) {
          VLOG(6) << "need trans from " << in_place << " to "
                  << kernel_key.backend();
          // build memcopy op
          auto out_place = phi::TransToPhiPlace(dst_backend);
          auto new_in_alloc_type =
              new_in_type.dyn_cast<dialect::AllocatedSelectedRowsType>();
          auto out_type = dialect::AllocatedSelectedRowsType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.dims(),
              new_in_alloc_type.data_layout(),
              new_in_alloc_type.lod(),
              new_in_alloc_type.offset());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "only support allocated dense tensor type for now"));
      }
    }
    vec_inputs.push_back(new_in);
  }

  return vec_inputs;
}

void AddShadowFeed(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Operation* kernel_op,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair) {
  bool feed_op_add_shadow_feed = (op_item->isa<paddle::dialect::FeedOp>()) &&
                                 platform::is_gpu_place(place);
  bool data_op_add_shadow_feed =
      (op_item->isa<paddle::dialect::DataOp>()) &&
      platform::is_gpu_place(place) &&
      (kernel_op->attributes()
           .at("place")
           .dyn_cast<dialect::PlaceAttribute>()
           .data()
           .GetType() == phi::AllocationType::UNDEFINED);
  bool add_shadow_feed = feed_op_add_shadow_feed || data_op_add_shadow_feed;
  if (add_shadow_feed) {
    // if shadow data op place not gpu,add shadow feed op
    phi::KernelKey shadow_key{
        phi::Backend::GPU,
        phi::DataLayout::ANY,
        TransToPhiDataType(
            op_item->result(0).type().dyn_cast<DenseTensorType>().dtype())};
    std::unordered_map<std::string, pir::Attribute> attr_map{
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.shadow_feed")},
        {"kernel_name", pir::StrAttribute::get(ctx, "shadow_feed")},
        {"kernel_key", dialect::KernelAttribute::get(ctx, shadow_key)}};

    auto out_type = paddle::dialect::AllocatedDenseTensorType::get(
        ctx,
        phi::TransToPhiPlace(shadow_key.backend()),
        op_item->result(0).type().dyn_cast<dialect::DenseTensorType>());

    pir::OpInfo phi_kernel_op_info =
        ctx->GetRegisteredOpInfo(paddle::dialect::PhiKernelOp::name());
    pir::Operation* shadow_op = pir::Operation::Create(
        {kernel_op->result(0)}, attr_map, {out_type}, phi_kernel_op_info);

    (*map_op_pair)[op_item] = shadow_op;
    block->push_back(shadow_op);
    if (op_item->num_results() > 0) {
      for (size_t i = 0; i < shadow_op->num_results(); ++i) {
        (*map_value_pair)[op_item->result(i)] = shadow_op->result(i);
      }
    }
  }
}

std::unique_ptr<OpYamlInfoParser> GetOpYamlInfoParser(pir::Operation* op) {
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();

  std::unique_ptr<OpYamlInfoParser> op_info_parser(nullptr);
  if (op_info_interface) {
    op_info_parser =
        std::make_unique<OpYamlInfoParser>(op_info_interface.GetOpInfo());
  }

  return op_info_parser;
}

std::string GetKernelFnStr(const OpYamlInfoParser* op_info_parser,
                           pir::Operation* op_item) {
  std::string kernel_fn_str;
  if (op_info_parser != nullptr) {
    kernel_fn_str = op_info_parser->OpRuntimeInfo().kernel_func;
  }

  if (op_item->isa<paddle::dialect::AddN_Op>() ||
      op_item->isa<paddle::dialect::AddNWithKernelOp>()) {
    if (op_item->result(0).type().isa<dialect::SelectedRowsType>()) {
      kernel_fn_str = "add_n_sr";
    }
  }
  return kernel_fn_str;
}

pir::Operation* BuildPhiKernelOp(
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const std::vector<pir::Value>& vec_inputs,
    const std::vector<pir::Type>& op_output_types,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair) {
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, op_item->name())},
      {"kernel_name", pir::StrAttribute::get(ctx, kernel_fn_str)},
      {"kernel_key", dialect::KernelAttribute::get(ctx, kernel_key)}};
  auto op_attr_map = op_item->attributes();

  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }

  if (op_item->HasTrait<paddle::dialect::InplaceTrait>()) {
    op_attribute.emplace("is_inplace", pir::BoolAttribute::get(ctx, true));
  }

  pir::OpInfo phi_kernel_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::PhiKernelOp::name());

  pir::OpInfo legacy_kernel_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::LegacyKernelOp::name());
  pir::Operation* op;
  if (dialect::IsLegacyOp(op_item->name())) {
    op = pir::Operation::Create(
        vec_inputs, op_attribute, op_output_types, legacy_kernel_op_info);
  } else {
    op = pir::Operation::Create(
        vec_inputs, op_attribute, op_output_types, phi_kernel_op_info);
  }

  (*map_op_pair)[op_item] = op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);

  return op;
}

void ProcessBlock(
    const phi::Place& place,
    pir::Block* block,
    pir::Block* new_block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::OpResult>* map_value_pair) {
  auto skip_feed_names = GetSkipFeedNames(block);

  for (auto op_item : *block) {
    VLOG(6) << "op name " << op_item->name();
    if ((op_item->isa<paddle::dialect::FeedOp>()) &&
        SkipFeedOp(op_item, skip_feed_names)) {
      continue;
    }

    // HandleSpecialOp
    if (SpecialLowerOps.count(op_item->name())) {
      HandleForSpecialOp(
          place, op_item, new_block, ctx, map_op_pair, map_value_pair);
      continue;
    }

    // Lower from PaddleDialect to KernelDialect

    auto op_info_parser = GetOpYamlInfoParser(op_item);

    auto kernel_fn_str = GetKernelFnStr(op_info_parser.get(), op_item);

    auto kernel_key = GetKernelKey(
        op_item, place, kernel_fn_str, *map_value_pair, op_info_parser.get());
    VLOG(6) << "kernel type " << kernel_key;

    // build output type
    auto op_output_types =
        BuildOpOutputType(op_item, kernel_fn_str, kernel_key, ctx);

    // build input
    auto vec_inputs = BuildOpInputList(op_item,
                                       kernel_fn_str,
                                       kernel_key,
                                       place,
                                       op_info_parser.get(),
                                       ctx,
                                       map_op_pair,
                                       map_value_pair,
                                       new_block);

    // build op
    pir::Operation* op = BuildPhiKernelOp(kernel_fn_str,
                                          kernel_key,
                                          vec_inputs,
                                          op_output_types,
                                          op_item,
                                          new_block,
                                          ctx,
                                          map_op_pair,
                                          map_value_pair);

    AddShadowFeed(
        place, op_item, op, new_block, ctx, map_op_pair, map_value_pair);
  }
}

std::unique_ptr<pir::Program> PdOpLowerToKernelPass(pir::Program* prog,
                                                    phi::Place place) {
  auto program = std::make_unique<pir::Program>(pir::IrContext::Instance());

  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

  std::unordered_map<pir::Operation*, pir::Operation*> map_op_pair;
  std::unordered_map<pir::Value, pir::OpResult> map_value_pair;

  ProcessBlock(
      place, block, program->block(), ctx, &map_op_pair, &map_value_pair);

  if (VLOG_IS_ON(2)) {
    std::stringstream ss1;
    program->Print(ss1);
    VLOG(2) << "Program after lowering to kernel pass : " << ss1.str();
  }
  return program;
}
}  // namespace dialect
}  // namespace paddle
