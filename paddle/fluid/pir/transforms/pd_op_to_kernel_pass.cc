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

#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"

#include <iostream>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/utils/flags.h"

PHI_DECLARE_bool(print_ir);
namespace paddle {
namespace dialect {

pir::Type ConvertOpTypeToKernelType(pir::IrContext* ctx,
                                    pir::Type op_type,
                                    phi::Place place) {
  if (op_type.isa<DenseTensorType>()) {
    return AllocatedDenseTensorType::get(
        ctx, place, op_type.dyn_cast<DenseTensorType>());
  } else if (op_type.isa<DenseTensorArrayType>()) {
    return AllocatedDenseTensorArrayType::get(
        ctx, place, op_type.dyn_cast<DenseTensorArrayType>());
  } else if (op_type.isa<SelectedRowsType>()) {
    return AllocatedSelectedRowsType::get(
        ctx, place, op_type.dyn_cast<SelectedRowsType>());
  }
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not support op type %s in ConvertOpTypeToKernelType.", op_type));
}

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
    pir::CombineOp::name(),
    pir::SliceOp::name(),
    pir::SplitOp::name(),
    pir::ConstantTensorOp::name(),
    pir::SetParameterOp::name(),
    pir::ParameterOp::name(),
    pir::ShadowOutputOp::name(),
    FeedOp::name(),
    DataOp::name(),
    ArrayLengthOp::name(),
    "cinn_runtime.jit_kernel",
};
const std::unordered_set<std::string> SpecialLowerOps = {
    pir::CombineOp::name(),
    pir::ConstantTensorOp::name(),
    pir::SetParameterOp::name(),
    pir::ParameterOp::name(),
    pir::ShadowOutputOp::name(),
    pir::SliceOp::name(),
    pir::SplitOp::name(),
    pir::YieldOp::name(),
    IfOp::name(),
    WhileOp::name(),
    pir::StackCreateOp::name(),
    pir::TuplePushOp::name(),
    pir::TuplePopOp::name(),
    HasElementsOp::name(),
    SelectInputOp::name(),
    "cinn_runtime.jit_kernel"};

static bool NeedFallBackCpu(const pir::Operation* op,
                            const std::string& kernel,
                            const phi::KernelKey& kernel_key) {
  if (UnchangeOutputOps.count(op->name()) || kernel == "" ||
      phi::KernelFactory::Instance().HasKernel(kernel, kernel_key)) {
    return false;
  }

  phi::KernelKey copy_key = kernel_key;
  if (copy_key.backend() == phi::Backend::GPUDNN) {
    copy_key.set_backend(phi::Backend::GPU);
    if (phi::KernelFactory::Instance().HasKernel(kernel, copy_key)) {
      return false;
    }
  }
  copy_key.set_backend(phi::Backend::CPU);
  if (phi::KernelFactory::Instance().HasKernel(kernel, copy_key)) {
    return true;
  }

  return false;
}

static bool NeedFallBackFromGPUDNN2GPU(pir::Operation* op,
                                       const phi::KernelKey kernel_key) {
  // NOTE(phlrain): keep the same kernel select strategy with
  // GetExepectKernelKey
  if (op->isa<Pool2dOp>() || op->isa<Pool2dGradOp>() || op->isa<Pool3dOp>() ||
      op->isa<Pool3dGradOp>()) {
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

static phi::Backend DeriveBackend(const std::string& op,
                                  const phi::Place& place,
                                  const OpYamlInfoParser* op_info_parser,
                                  phi::Backend kernel_backend,
                                  size_t input_index) {
  // NOTE: Paramters are initilizered on executor place defined
  if ((op.compare(pir::SetParameterOp::name()) == 0 ||
       op.compare(pir::ShadowOutputOp::name()) == 0) &&
      place.GetType() == phi::AllocationType::GPU) {
    return phi::TransToPhiBackend(place);
  }
  // Tensor Attribute should on cpu backend for better performance
  if (op_info_parser != nullptr &&
      op_info_parser->IsTensorAttribute(input_index)) {
    return phi::Backend::CPU;
  }
  return kernel_backend;
}

static phi::Backend ChooseInputBackend(const phi::Kernel& kernel,
                                       size_t input_index,
                                       phi::Backend default_backend) {
  if (kernel.GetKernelRegisteredType() == phi::KernelRegisteredType::FUNCTION) {
    return kernel.InputAt(input_index).backend;
  }
  return default_backend;
}

static std::set<std::string> GetInputsByDataOp(pir::Block* block) {
  std::set<std::string> data_op_names;
  for (auto& op_item : *block) {
    if (op_item.isa<DataOp>()) {
      data_op_names.insert(op_item.attributes()
                               .at("name")
                               .dyn_cast<pir::StrAttribute>()
                               .AsString());
    }
  }
  return data_op_names;
}

template <class IrType>
static phi::DenseTensorMeta parse_tensor_meta(IrType type) {
  auto dtype = TransToPhiDataType(type.dtype());
  return phi::DenseTensorMeta(
      dtype, type.dims(), type.data_layout(), type.lod(), type.offset());
}

static std::vector<std::shared_ptr<phi::TensorBase>> PrepareFakeTensors(
    pir::Value input) {
  std::vector<std::shared_ptr<phi::TensorBase>> res;
  auto in_type = input.type();

  auto fake_dt = [](const AllocatedDenseTensorType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedDenseTensorType>(type);
    return std::make_shared<phi::DenseTensor>(holder, meta);
  };

  auto fake_sr = [](const AllocatedSelectedRowsType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    phi::DenseTensorMeta meta =
        parse_tensor_meta<AllocatedSelectedRowsType>(type);

    std::vector<int64_t> rows;
    rows.clear();
    auto sr = std::make_shared<phi::SelectedRows>(rows, 0);
    phi::DenseTensor dense_tensor(holder, meta);
    *(sr->mutable_value()) = dense_tensor;

    return sr;
  };

  auto fake_tensor_array = [](const AllocatedDenseTensorArrayType& type) {
    auto ptr = new phi::Allocation(nullptr, 0, type.place());
    std::shared_ptr<phi::Allocation> holder(ptr);
    auto dtype = TransToPhiDataType(type.dtype());
    phi::DenseTensorMeta meta(dtype, {});
    phi::DenseTensor dt(holder, meta);
    auto tensor_array = std::make_shared<phi::TensorArray>(0);
    tensor_array->set_type(dtype);
    return tensor_array;
  };

  if (in_type.isa<AllocatedDenseTensorType>()) {
    res.push_back(fake_dt(in_type.dyn_cast<AllocatedDenseTensorType>()));
  } else if (in_type.isa<AllocatedSelectedRowsType>()) {
    res.push_back(fake_sr(in_type.dyn_cast<AllocatedSelectedRowsType>()));
  } else if (in_type.isa<pir::VectorType>()) {
    auto inner_types = in_type.dyn_cast<pir::VectorType>().data();
    for (size_t i = 0; i < inner_types.size(); ++i) {
      if (inner_types[i].isa<AllocatedDenseTensorType>()) {
        res.push_back(
            fake_dt(inner_types[i].dyn_cast<AllocatedDenseTensorType>()));
      } else if (inner_types[i].isa<AllocatedSelectedRowsType>()) {
        res.push_back(
            fake_sr(inner_types[i].dyn_cast<AllocatedSelectedRowsType>()));
      }
    }
  } else if (in_type.isa<AllocatedDenseTensorArrayType>()) {
    res.push_back(
        fake_tensor_array(in_type.dyn_cast<AllocatedDenseTensorArrayType>()));
  }
  return res;
}

static pir::OpResult AddPlaceTransferOp(pir::Value in,
                                        pir::Type out_type,
                                        const phi::Place& src_place,
                                        const phi::Place& dst_place,
                                        const phi::KernelKey& kernel_key,
                                        pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  auto copy_kernel_key = kernel_key;
  std::unordered_map<std::string, pir::Attribute> op_attribute;
  if ((src_place.GetType() == phi::AllocationType::CPU) &&
      (dst_place.GetType() == phi::AllocationType::GPU)) {
    copy_kernel_key.set_backend(phi::Backend::GPU);
    op_attribute = {
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.memcpy_h2d")},
        {"kernel_name", pir::StrAttribute::get(ctx, "memcpy_h2d")},
        {"kernel_key", KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 1)}};
  } else if ((src_place.GetType() == phi::AllocationType::GPU) &&
             (dst_place.GetType() == phi::AllocationType::CPU)) {
    copy_kernel_key.set_backend(phi::Backend::GPU);
    op_attribute = {
        {"op_name", pir::StrAttribute::get(ctx, "pd_op.memcpy_d2h")},
        {"kernel_name", pir::StrAttribute::get(ctx, "memcpy_d2h")},
        {"kernel_key", KernelAttribute::get(ctx, copy_kernel_key)},
        {"dst_place_type", pir::Int32Attribute::get(ctx, 0)}};
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support cpu to gpu and gpu to cpu"));
  }

  pir::OpInfo kernel_op_info = ctx->GetRegisteredOpInfo(PhiKernelOp::name());
  pir::Operation* op =
      pir::Operation::Create({in}, op_attribute, {out_type}, kernel_op_info);
  auto in_op = in.dyn_cast<pir::OpResult>().owner();
  if (in_op && in_op->HasAttribute(kAttrIsPersisable)) {
    op->set_attribute(kAttrIsPersisable, in_op->attribute(kAttrIsPersisable));
  }
  block->push_back(op);
  auto new_in = op->result(0);
  return new_in;
}

static bool NeedTransformDataType(const phi::DataType& l,
                                  const phi::DataType& r) {
  return l != phi::DataType::ALL_DTYPE && r != phi::DataType::ALL_DTYPE &&
         l != r;
}

static const phi::DataType GetKernelTypeforVar(
    pir::Operation* op,
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::KernelKey* expected_kernel_key) {
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op->name());
  auto get_kernel_type_for_var =
      op_info.GetInterfaceImpl<GetKernelTypeForVarInterface>();
  if (get_kernel_type_for_var) {
    phi::DataType kernel_dtype_for_var =
        get_kernel_type_for_var->get_kernel_type_for_var_(
            var_name, tensor_dtype, (*expected_kernel_key).dtype());
    return kernel_dtype_for_var;
  }
  return (*expected_kernel_key).dtype();
}

template <class IrType>
std::tuple<phi::Backend, phi::DataLayout> parse_kernel_info(pir::Type type) {
  phi::Backend backend =
      paddle::experimental::ParseBackend(type.dyn_cast<IrType>().place());
  phi::DataLayout layout =
      paddle::experimental::ParseLayout(type.dyn_cast<IrType>().data_layout());
  return {backend, layout};
}

template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place& place,
                             pir::Type out_dtype,
                             pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

static pir::Type BuildDtypeTransferOutputType(pir::Type type,
                                              const phi::Place& place,
                                              phi::DataType data_dtype,
                                              pir::IrContext* ctx) {
  if (type.isa<AllocatedDenseTensorType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_type<AllocatedDenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    auto out_dtype = TransToIrDataType(data_dtype, ctx);
    return create_type<AllocatedSelectedRowsType, AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType and SelectedRowsType"));
  }
}

static pir::Type BuildOutputType(pir::Type type,
                                 const phi::Place& place,
                                 pir::IrContext* ctx) {
  if (type.isa<DenseTensorType>()) {
    auto out_dtype = type.dyn_cast<DenseTensorType>().dtype();
    return create_type<DenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SelectedRowsType>()) {
    auto out_dtype = type.dyn_cast<SelectedRowsType>().dtype();
    return create_type<SelectedRowsType, AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<DenseTensorArrayType>()) {
    auto array_type = type.dyn_cast<DenseTensorArrayType>();
    return AllocatedDenseTensorArrayType::get(
        ctx, place, array_type.dtype(), array_type.data_layout());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType and SelectedRowsType"));
  }
}

pir::OpResult AddDtypeTransferOp(pir::Value in,
                                 pir::Block* block,
                                 const phi::KernelKey& kernel_key,
                                 const phi::Place& origin_place,
                                 const phi::Place& out_place,
                                 const phi::DataType& src_dtype,
                                 const phi::DataType& dst_dtype) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  pir::OpInfo kernel_op_info = ctx->GetRegisteredOpInfo(PhiKernelOp::name());

  // Get kernelkey (backend、layout)
  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;

  if (in.type().isa<AllocatedDenseTensorType>()) {
    auto out = parse_kernel_info<AllocatedDenseTensorType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else if (in.type().isa<AllocatedSelectedRowsType>()) {
    auto out = parse_kernel_info<AllocatedSelectedRowsType>(in.type());
    kernel_backend = std::get<0>(out);
    kernel_layout = std::get<1>(out);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Get kernelkey for CastOp only support "
                                   "DenseTensorType and SelectedRowsType"));
  }

  if (kernel_backend == phi::Backend::UNDEFINED) {
    kernel_backend = paddle::experimental::ParseBackend(origin_place);
  }

  phi::KernelKey cast_kernel_key(kernel_backend, kernel_layout, src_dtype);

  // Create CastOp
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, "pd_op.cast")},
      {"kernel_name", pir::StrAttribute::get(ctx, "cast")},
      {"kernel_key", KernelAttribute::get(ctx, cast_kernel_key)},
      {"dtype", DataTypeAttribute::get(ctx, dst_dtype)}};

  pir::Type output_types =
      BuildDtypeTransferOutputType(in.type(), out_place, dst_dtype, ctx);

  pir::Operation* op = pir::Operation::Create(
      {in}, op_attribute, {output_types}, kernel_op_info);

  auto in_op = in.dyn_cast<pir::OpResult>().owner();
  if (in_op && in_op->HasAttribute(kAttrIsPersisable)) {
    op->set_attribute(kAttrIsPersisable, in_op->attribute(kAttrIsPersisable));
  }
  block->push_back(op);
  pir::OpResult new_in = op->result(0);
  return new_in;
}

static phi::DataType GetKernelDtypeByYaml(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const OpYamlInfoParser* op_info_parser) {
  auto& attr_map = op->attributes();
  auto& data_type_info = op_info_parser->OpRuntimeInfo().kernel_key_dtype;
  phi::DataType kernel_data_type = phi::DataType::UNDEFINED;

  for (size_t i = 0; i < data_type_info.size(); ++i) {
    auto slot_name = data_type_info[i];
    auto& input_map = op_info_parser->InputName2Id();

    bool is_complex_tag = false;
    if (slot_name.find("complex:") == 0) {
      slot_name = slot_name.substr(8);
      is_complex_tag = true;
    }

    auto find_it = Str2PhiDataType.find(slot_name);
    if (find_it != Str2PhiDataType.end()) {
      kernel_data_type = find_it->second;
    } else if (input_map.count(slot_name)) {
      // parse from input
      int in_index = static_cast<int>(input_map.at(slot_name));
      auto type = map_value_pair.at(op->operand_source(in_index)).type();

      if (type.isa<AllocatedDenseTensorType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedDenseTensorType>().dtype());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_data_type = phi::DataType::UNDEFINED;
        } else {
          if (vec_data[0].isa<AllocatedDenseTensorType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedDenseTensorType>().dtype());
          } else if (vec_data[0].isa<AllocatedSelectedRowsType>()) {
            kernel_data_type = TransToPhiDataType(
                vec_data[0].dyn_cast<AllocatedSelectedRowsType>().dtype());
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "Only support DenseTensorType and SelectedRowsType in vector"));
          }
        }
      } else if (type.isa<AllocatedSelectedRowsType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedSelectedRowsType>().dtype());
      } else if (type.isa<AllocatedDenseTensorArrayType>()) {
        kernel_data_type = TransToPhiDataType(
            type.dyn_cast<AllocatedDenseTensorArrayType>().dtype());
      } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "Only support DenseTensorType, SelectedRows, VectorType"));
      }
      if (is_complex_tag) {
        kernel_data_type = phi::dtype::ToComplex(kernel_data_type);
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
      kernel_data_type =
          attr_map.at(slot_name).dyn_cast<DataTypeAttribute>().data();
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

static phi::Backend GetKernelBackendByYaml(
    const pir::Operation* op,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const OpYamlInfoParser* op_info_parser,
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

      if (type.isa<AllocatedDenseTensorType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedDenseTensorType>().place());
      } else if (type.isa<pir::VectorType>()) {
        auto vec_data = type.dyn_cast<pir::VectorType>().data();
        if (vec_data.empty()) {
          kernel_backend = phi::Backend::UNDEFINED;
        } else {
          if (vec_data[0].isa<AllocatedDenseTensorType>()) {
            kernel_backend = paddle::experimental::ParseBackend(
                vec_data[0].dyn_cast<AllocatedDenseTensorType>().place());
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "Only support DenseTensorType in vector"));
          }
        }
      } else if (type.isa<AllocatedSelectedRowsType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedSelectedRowsType>().place());
      } else if (type.isa<AllocatedDenseTensorArrayType>()) {
        kernel_backend = paddle::experimental::ParseBackend(
            type.dyn_cast<AllocatedDenseTensorArrayType>().place());
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
          attr_map.at(slot_name).dyn_cast<PlaceAttribute>().data());
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

std::unique_ptr<OpYamlInfoParser> GetOpYamlInfoParser(pir::Operation* op) {
  OpYamlInfoInterface op_info_interface = op->dyn_cast<OpYamlInfoInterface>();

  std::unique_ptr<OpYamlInfoParser> op_info_parser(nullptr);
  if (op_info_interface) {
    op_info_parser = std::make_unique<OpYamlInfoParser>(
        op_info_interface.GetOpInfo(), IsLegacyOp(op->name()));
  }

  return op_info_parser;
}

std::string GetKernelName(const OpYamlInfoParser* op_info_parser,
                          pir::Operation* op_item) {
  std::string kernel_fn_str;
  if (op_info_parser != nullptr) {
    kernel_fn_str = op_info_parser->OpRuntimeInfo().kernel_func;
  }

  if (op_item->isa<AddN_Op>() || op_item->isa<AddNWithKernelOp>()) {
    if (op_item->result(0).type().isa<SelectedRowsType>()) {
      kernel_fn_str = "add_n_sr";
    }
  }
  return kernel_fn_str;
}

phi::KernelKey GetKernelKey(
    pir::Operation* op,
    const phi::Place& place,
    const std::string& kernel_fn_str,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    OpYamlInfoParser* op_info_parser = nullptr) {
  if (op->isa<FeedOp>() || op->isa<FetchOp>() || op->isa<ArrayLengthOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    VLOG(6) << "FeedOp doesn't need a kernel. Backend: CPU, DataLayout: ANY";
    return {phi::Backend::CPU,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->isa<DataOp>()) {
    // NOTE, for now feed op don't need a kernel, so the data type from Op
    // Result the next op use base program datatype
    VLOG(6) << "DataOp doesn't need a kernel";
    auto data_place =
        op->attributes().at("place").dyn_cast<PlaceAttribute>().data();

    auto backend = paddle::experimental::ParseBackend(data_place);

    return {backend,
            phi::DataLayout::ANY,
            TransToPhiDataType(
                op->result(0).type().dyn_cast<DenseTensorType>().dtype())};
  }

  if (op->isa<SeedOp>()) {
    VLOG(6) << "SeedOp doesn't need a kernel";
    auto backend = paddle::experimental::ParseBackend(place);
    return {backend, phi::DataLayout::ANY, phi::DataType::INT32};
  }

  if (op->isa<FullWithTensorOp>()) {
    VLOG(6) << "FullWithTensorOp doesn't need a kernel";
    auto backend = paddle::experimental::ParseBackend(place);
    auto dtype =
        op->attributes().at("dtype").dyn_cast<DataTypeAttribute>().data();

    return {backend, phi::DataLayout::ANY, dtype};
  }

  phi::Backend kernel_backend = phi::Backend::UNDEFINED;
  phi::DataLayout kernel_layout = phi::DataLayout::UNDEFINED;
  phi::DataType kernel_dtype = phi::DataType::UNDEFINED;

  if (op_info_parser != nullptr) {
    // only suppurt non vector input for now
    int tensor_input_number =
        static_cast<int>(op_info_parser->InputTensorNumber());
    VLOG(8) << "Begin to infer kernel key from op_info_parser(defined by yaml "
               "info)";
    // get datatype info
    kernel_dtype = GetKernelDtypeByYaml(op, map_value_pair, op_info_parser);
    VLOG(8) << "Infer kernel data_type: [" << kernel_dtype
            << "] from yaml info";
    kernel_backend =
        GetKernelBackendByYaml(op, map_value_pair, op_info_parser, place);
    VLOG(8) << "Infer kernel backend: [" << kernel_backend
            << "] from yaml info";
    // parse all the input tensor
    if (tensor_input_number == 0 || op->isa<Full_Op>()) {
      // all the information have to get from attribute and context
      if (kernel_backend == phi::Backend::UNDEFINED) {
        kernel_backend = paddle::experimental::ParseBackend(place);
        VLOG(8) << "Infer kernel backend: [" << kernel_backend
                << "] when tensor_input_number == 0  or is Full_Op";
      }
    }
  }

  // TODO(zhangbo): Add ParseKernelInterface
  ParseKernelKeyInterface parse_kernel_key_interface =
      op->dyn_cast<ParseKernelKeyInterface>();
  if (parse_kernel_key_interface) {
    auto parsed_key = parse_kernel_key_interface.ParseKernelKey(op);
    kernel_dtype = std::get<0>(parsed_key);
    kernel_backend = std::get<1>(parsed_key);
  }

  if ((kernel_backend == phi::Backend::UNDEFINED ||
       kernel_dtype == phi::DataType::UNDEFINED) &&
      op->num_operands() > 0) {
    paddle::experimental::detail::KernelKeyParser kernel_key_parser;
    VLOG(8) << "Begin to infer kernel key from op operands";
    for (size_t i = 0; i < op->num_operands(); ++i) {
      // NOTE, only op with OpYamlInfo can have TensorArr
      if (op_info_parser != nullptr && op_info_parser->IsTensorAttribute(i)) {
        VLOG(8) << "input (" << i << ") doesn't have TensorArr";
        continue;
      }
      auto input_tmp = op->operand_source(i);
      // NOTE: if not input_tmp, it's an optional input
      if (!input_tmp) {
        VLOG(8) << "input (" << i << ") is NULL (optional input)";
        continue;
      }
      auto new_input_tmp = map_value_pair.at(input_tmp);

      auto fake_tensors = PrepareFakeTensors(new_input_tmp);
      for (auto& fake_tensor : fake_tensors) {
        kernel_key_parser.AssignKernelKeySet(*fake_tensor);
      }

      // Because we can't make sure the place when build data op
      // and the output place of data op is undefined. It means we
      // don't know how to select the kernel in the next of op that
      // uses data op outout as inputs. So, we need set kernel backend
      // manually.
      auto op_res = input_tmp.dyn_cast<pir::OpResult>();
      if (!op_res) {
        continue;
      }
      if (op_res.owner()->isa<DataOp>()) {
        auto data_op = op->operand_source(i).dyn_cast<pir::OpResult>().owner();
        auto data_place = data_op->attribute<PlaceAttribute>("place").data();

        auto data_op_backend = paddle::experimental::ParseBackend(data_place);
        if (data_op_backend == phi::Backend::UNDEFINED) {
          data_op_backend = paddle::experimental::ParseBackend(place);
        }
        kernel_key_parser.key_set.backend_set =
            kernel_key_parser.key_set.backend_set |
            paddle::experimental::BackendSet(data_op_backend);
        VLOG(8) << "Update kernel backend set from owner op (DataOp): "
                << data_op_backend;
      } else if (op_res.owner()->isa<pir::CombineOp>()) {
        auto combine_op = op_res.owner();
        for (size_t j = 0; j < combine_op->num_operands(); ++j) {
          auto combine_op_res =
              combine_op->operand_source(j).dyn_cast<pir::OpResult>();
          if (!combine_op_res) {
            continue;
          }
          if (combine_op_res.owner()->isa<DataOp>()) {
            auto data_op = combine_op_res.owner();
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
            VLOG(8) << "Update kernel backend set from owner op (CombineOp): "
                    << data_op_backend;
            break;
          }
        }
      }
    }

    auto kernel_key_set = kernel_key_parser.key_set;

    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

    if (kernel_backend == phi::Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
      if (kernel_backend != phi::Backend::UNDEFINED) {
        VLOG(8) << "Infer kernel backend from op operands";
      }
    }
    if (kernel_layout == phi::DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
      if (kernel_layout != phi::DataLayout::UNDEFINED) {
        VLOG(8) << "Infer kernel layout from op operands";
      }
    }
    if (kernel_dtype == phi::DataType::UNDEFINED) {
      kernel_dtype = kernel_key.dtype();
      if (kernel_dtype != phi::DataType::UNDEFINED) {
        VLOG(8) << "Infer kernel data_type from op operands";
      }
    }
  }

  if (kernel_backend == phi::Backend::UNDEFINED) {
    VLOG(8) << "Kernel backend cannot be infered from op operands";
    kernel_backend = paddle::experimental::ParseBackend(place);
  }

  phi::KernelKey res(kernel_backend, kernel_layout, kernel_dtype);

  // kernel backend infered incorrectly from memcpy op operands,
  // case that place from (not GPU) to GPU.
  // We handle this special case by following code to fix up the problem.
  // This could be further improved if we had another method.
  if (!platform::is_gpu_place(place)) {
    if (op->isa<MemcpyOp>()) {
      VLOG(6) << "MemcpyOp need a special handle";
      int dst_place_type = op->attribute("dst_place_type")
                               .dyn_cast<pir::Int32Attribute>()
                               .data();
      if (dst_place_type == 1) {
        res.set_backend(phi::Backend::GPU);
      }
    }
  }

  if (op->isa<LoadCombineOp>()) {
    res.set_dtype(phi::DataType::FLOAT32);
    VLOG(8) << "LoadCombineOp's kernel data type must be FLOAT32";
  }

  if (op->isa<CSyncCommStream_Op>() || op->isa<CSyncCommStreamOp>()) {
    res.set_dtype(phi::DataType::FLOAT32);
    VLOG(8) << "CSyncCommStream_Op/CSyncCommStreamOp's kernel data type must "
               "be FLOAT32";
  }

  if (NeedFallBackCpu((op), kernel_fn_str, res)) {
    res.set_backend(phi::Backend::CPU);
    VLOG(8) << "kernel backend must be on CPU when need fallback";
  }

  if (NeedFallBackFromGPUDNN2GPU(op, res)) {
    res.set_backend(phi::Backend::GPU);
    VLOG(8) << "kernel backend must be on GPU when need fallback from GPUDNN "
               "to GPU";
  }

  return res;
}

void HandleForIfOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  auto old_cond = op_item->operand_source(0);

  PADDLE_ENFORCE_EQ(
      map_value_pair->count(old_cond),
      true,
      phi::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
  auto new_cond = map_value_pair->at(old_cond);

  // NOTE(zhangbo): IfOp's input cond should be a cpu type.
  AllocatedDenseTensorType new_cond_type =
      new_cond.type().dyn_cast<AllocatedDenseTensorType>();
  if (new_cond_type) {
    if (new_cond_type.place().GetType() == phi::AllocationType::GPU) {
      auto out_type = AllocatedDenseTensorType::get(
          ctx, phi::CPUPlace(), old_cond.type().dyn_cast<DenseTensorType>());
      phi::KernelKey kernel_key(
          phi::Backend::GPU, phi::DataLayout::ALL_LAYOUT, phi::DataType::BOOL);
      new_cond = AddPlaceTransferOp(new_cond,
                                    out_type,
                                    new_cond_type.place(),
                                    phi::CPUPlace(),
                                    kernel_key,
                                    block);
    }
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("IfOp onlu support DenseTensorType"));
  }

  // Create IfOp and insert to kernel dialect program
  pir::Builder builder(ctx, block);
  auto old_ifop = op_item->dyn_cast<IfOp>();
  std::vector<pir::Type> new_ifop_outputs;
  for (size_t i = 0; i < old_ifop.num_results(); ++i) {
    new_ifop_outputs.push_back(
        ConvertOpTypeToKernelType(ctx, old_ifop.result(i).type(), place));
  }
  auto new_ifop = builder.Build<IfOp>(new_cond, std::move(new_ifop_outputs));

  // process true block
  auto& true_block = new_ifop.true_block();
  ProcessBlock(place,
               &old_ifop.true_block(),
               &true_block,
               ctx,
               map_op_pair,
               map_value_pair);

  // process false block
  auto& false_block = new_ifop.false_block();
  ProcessBlock(place,
               &old_ifop.false_block(),
               &false_block,
               ctx,
               map_op_pair,
               map_value_pair);

  // update map
  (*map_op_pair)[op_item] = new_ifop;
  for (size_t i = 0; i < op_item->num_results(); ++i) {
    (*map_value_pair)[op_item->result(i)] = new_ifop->result(i);
  }
}

void HandleForWhileOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  std::vector<pir::Value> vec_in;
  pir::Value cond_val;
  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);

    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        phi::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", 0, op_item->name()));
    auto new_in = map_value_pair->at(cur_in);
    if (i == 0) {
      cond_val = new_in;
    } else {
      vec_in.push_back(new_in);
    }
  }

  pir::Builder builder(ctx, block);
  auto base_while_op = op_item->dyn_cast<WhileOp>();
  auto new_while_op = builder.Build<WhileOp>(cond_val, vec_in);
  pir::Block& body_block = new_while_op.body();
  for (size_t i = 0; i < vec_in.size(); ++i) {
    auto block_arg = body_block.arg(i);
    (*map_value_pair)[base_while_op.body().arg(i)] = block_arg;
  }

  // process body block
  ProcessBlock(place,
               &base_while_op.body(),
               &body_block,
               ctx,
               map_op_pair,
               map_value_pair);

  (*map_op_pair)[op_item] = new_while_op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = new_while_op->result(i);
    }
  }
}

pir::Value GetNewInput(
    const pir::Value cur_in,
    const std::unordered_map<pir::Value, pir::Value>& map_value_pair,
    const int index,
    const std::string& op_name) {
  PADDLE_ENFORCE_EQ(
      map_value_pair.count(cur_in),
      true,
      phi::errors::PreconditionNotMet(
          "[%d]'s input of [%s] op MUST be in map pair", index, op_name));
  auto new_in = map_value_pair.at(cur_in);
  return new_in;
}

phi::Place ParsePhiPlace(pir::Type type) {
  if (type.isa<AllocatedDenseTensorType>()) {
    return type.dyn_cast<AllocatedDenseTensorType>().place();
  } else if (type.isa<AllocatedSelectedRowsType>()) {
    return type.dyn_cast<AllocatedSelectedRowsType>().place();
  } else if (type.isa<AllocatedDenseTensorArrayType>()) {
    return type.dyn_cast<AllocatedDenseTensorArrayType>().place();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "ParsePhiPlace only support AllocatedDenseTensorType or "
        "AllocatedSelectedRowsType or AllocatedDenseTensorArrayType"));
  }
}

void HandleForSpecialOp(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  if (op_item->isa<IfOp>()) {
    HandleForIfOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
    return;
  }

  if (op_item->isa<WhileOp>()) {
    HandleForWhileOp(place, op_item, block, ctx, map_op_pair, map_value_pair);
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
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);
        vec_inner_types.push_back(new_in.type());
      }
    }
    // Copy op output type

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types.push_back(t1);
  }

  if (op_item->isa<::pir::ParameterOp>()) {
    op_output_types.push_back(
        BuildOutputType(op_item->result(0).type(), place, ctx));
  }

  if (op_item->isa<::pir::ConstantTensorOp>()) {
    op_output_types.push_back(
        BuildOutputType(op_item->result(0).type(), phi::CPUPlace(), ctx));
  }

  if (op_item->isa<::pir::SliceOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
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
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
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

  if (op_item->isa<::pir::YieldOp>() || op_item->isa<::pir::ShadowOutputOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);
      }
    }
  }

  if (op_item->isa<::pir::SetParameterOp>()) {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        // NOTE(zhangbo): parameter place is equal to exe place.
        if (new_in.type().isa<AllocatedDenseTensorType>()) {
          auto in_place =
              new_in.type().dyn_cast<AllocatedDenseTensorType>().place();
          auto dst_backend = phi::TransToPhiBackend(place);
          bool need_trans =
              (in_place.GetType() != phi::AllocationType::UNDEFINED) &&
              (paddle::experimental::NeedTransformPlace(
                  in_place, dst_backend, {}));
          if (need_trans) {
            VLOG(6) << "need trans from " << in_place << " to " << dst_backend;
            // build memcopy op
            auto out_place = phi::TransToPhiPlace(dst_backend);
            auto new_in_alloc_type =
                new_in.type().dyn_cast<AllocatedDenseTensorType>();
            auto out_type =
                AllocatedDenseTensorType::get(ctx,
                                              out_place,
                                              new_in_alloc_type.dtype(),
                                              new_in_alloc_type.dims(),
                                              new_in_alloc_type.data_layout(),
                                              new_in_alloc_type.lod(),
                                              new_in_alloc_type.offset());
            auto op_info_parser = GetOpYamlInfoParser(op_item);
            auto kernel_name = GetKernelName(op_info_parser.get(), op_item);
            auto kernel_key = GetKernelKey(op_item,
                                           place,
                                           kernel_name,
                                           *map_value_pair,
                                           op_info_parser.get());
            VLOG(6) << "kernel type " << kernel_key;
            new_in = AddPlaceTransferOp(
                new_in, out_type, in_place, out_place, kernel_key, block);
          }
        }
        vec_inputs.push_back(new_in);
      }
    }
  }

  if (op_item->isa<::pir::StackCreateOp>() ||
      op_item->isa<::pir::TuplePushOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      if (!cur_in) {
        vec_inputs.emplace_back();
        continue;
      }
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(op_item->result(i).type());
    }
  }

  if (op_item->isa<HasElementsOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    PADDLE_ENFORCE_EQ(op_item->result(0).type().isa<DenseTensorType>(),
                      true,
                      phi::errors::PreconditionNotMet(
                          "HasElementsOp's output should be bool type"));
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(
          BuildOutputType(op_item->result(i).type(), phi::CPUPlace(), ctx));
    }
  }

  if (op_item->isa<::pir::TuplePopOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }

    auto pop_back_op = op_item->dyn_cast<::pir::TuplePopOp>();
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto cur_inlet_element = pop_back_op.inlet_element(i);
      PADDLE_ENFORCE_EQ(map_value_pair->count(cur_inlet_element),
                        true,
                        phi::errors::PreconditionNotMet(
                            "[%d]'s output of [%s] op MUST be in map pair",
                            i,
                            op_item->name()));
      auto new_inlet_element = map_value_pair->at(cur_inlet_element);

      op_output_types.push_back(new_inlet_element.type());
    }
  }

  if (op_item->isa<SelectInputOp>()) {
    for (size_t i = 0; i < op_item->num_operands(); ++i) {
      auto cur_in = op_item->operand_source(i);
      auto new_in = GetNewInput(
          cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
      vec_inputs.push_back(new_in);
    }
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(vec_inputs[1].type());
    }
  }

  if (op_item->name() == "cinn_runtime.jit_kernel") {
    if (op_item->num_operands() > 0) {
      for (size_t i = 0; i < op_item->num_operands(); ++i) {
        auto cur_in = op_item->operand_source(i);
        if (!cur_in) {
          vec_inputs.emplace_back();
          continue;
        }
        auto new_in = GetNewInput(
            cur_in, *map_value_pair, static_cast<int>(i), op_item->name());
        vec_inputs.push_back(new_in);
      }
    }

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      op_output_types.push_back(AllocatedDenseTensorType::get(
          ctx, place, op_item->result(i).type().dyn_cast<DenseTensorType>()));
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
  VLOG(6) << "Deep copy a new special op: " << op_item->name();
}

std::vector<pir::Type> BuildOutputs(pir::Operation* op_item,
                                    const std::string& kernel_fn_str,
                                    const phi::KernelKey& kernel_key,
                                    pir::IrContext* ctx) {
  if (op_item->num_results() == 0) {
    return {};
  }
  std::vector<pir::Type> op_output_types;

  auto phi_kernel = phi::KernelFactory::Instance().SelectKernelWithGPUDNN(
      kernel_fn_str, kernel_key);
  VLOG(6) << "[" << kernel_fn_str
          << "] selected kernel(is_valid: " << phi_kernel.IsValid()
          << " ): " << kernel_key;

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
    if ((!UnchangeOutputOps.count(op_item->name())) &&
        (!IsLegacyOp(op_item->name())) && phi_kernel.IsValid()) {
      out_place = phi::TransToPhiPlace(output_defs[i].backend);
    }

    auto result_type = op_item->result(i).type();
    if (!result_type) {
      op_output_types.push_back(result_type);
    } else if (result_type.isa<DenseTensorType>() ||
               result_type.isa<SelectedRowsType>() ||
               result_type.isa<DenseTensorArrayType>()) {
      op_output_types.push_back(BuildOutputType(result_type, out_place, ctx));
    } else if (result_type.isa<pir::VectorType>()) {
      std::vector<pir::Type> vec_inner_types;
      auto base_types = result_type.dyn_cast<pir::VectorType>().data();
      for (auto& base_type : base_types) {
        if (base_type) {
          if (base_type.isa<DenseTensorType>() ||
              base_type.isa<SelectedRowsType>()) {
            vec_inner_types.push_back(
                BuildOutputType(base_type, out_place, ctx));
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
          auto dense_tensor_dtype = DenseTensorType::get(
              ctx, fp32_dtype, dims, data_layout, lod, offset);
          auto allocated_dense_tensor_dtype =
              AllocatedDenseTensorType::get(ctx, out_place, dense_tensor_dtype);
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

std::vector<pir::Value> BuildInputs(
    pir::Operation* op_item,
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const phi::Place place,
    const OpYamlInfoParser* op_info_parser,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair,
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

    int tensor_param_index = i;
    if (kernel.IsValid()) {
      tensor_param_index = op_info_parser->GetTensorParamIndexByArgsName(
          op_info_parser->InputNames()[i]);
      // the input of op args is not the kernel parameter
      if (tensor_param_index == -1) {
        vec_inputs.emplace_back(new_in);
        continue;
      }
    }

    // 1.backend transfer
    bool check_place_transfer =
        (op_item->isa<::pir::SetParameterOp>()) ||
        (kernel.IsValid() && (!UnchangeOutputOps.count(op_item->name())));

    if (check_place_transfer) {
      if (new_in_type.isa<AllocatedDenseTensorType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedDenseTensorType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op "
                << op_item->name();

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
              new_in_type.dyn_cast<AllocatedDenseTensorType>();
          auto out_type =
              AllocatedDenseTensorType::get(ctx,
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
            if (in_i_type.isa<AllocatedDenseTensorType>()) {
              place = in_i_type.dyn_cast<AllocatedDenseTensorType>().place();
            } else if (in_i_type.isa<AllocatedSelectedRowsType>()) {
              place = in_i_type.dyn_cast<AllocatedSelectedRowsType>().place();
            } else {
              PADDLE_THROW(phi::errors::Unimplemented(
                  "builtin.combine Input type only support "
                  "VectorType<DenseTensorType> and "
                  "VectorType<SelectedRowsType>"));
            }

            // get input args def type
            auto args_def = kernel.args_def();
            auto input_defs = args_def.input_defs();

            auto input_backend = ChooseInputBackend(
                kernel, tensor_param_index, kernel_key.backend());
            bool need_trans =
                (place.GetType() != phi::AllocationType::UNDEFINED) &&
                (op_info_parser != nullptr &&
                 !op_info_parser->IsTensorAttribute(i)) &&
                (paddle::experimental::NeedTransformPlace(
                    place, input_backend, {}));
            if (need_trans) {
              // build memcopy op
              auto out_place = phi::TransToPhiPlace(input_backend);
              pir::Type out_type;
              if (in_i_type.isa<AllocatedDenseTensorType>()) {
                out_type = AllocatedDenseTensorType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<DenseTensorType>());
              } else if (in_i_type.isa<AllocatedSelectedRowsType>()) {
                out_type = AllocatedSelectedRowsType::get(
                    ctx,
                    out_place,
                    pre_define_op->operand_source(j)
                        .type()
                        .dyn_cast<SelectedRowsType>());
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
      } else if (new_in_type.isa<AllocatedSelectedRowsType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedSelectedRowsType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
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
              new_in_type.dyn_cast<AllocatedSelectedRowsType>();
          auto out_type =
              AllocatedSelectedRowsType::get(ctx,
                                             out_place,
                                             new_in_alloc_type.dtype(),
                                             new_in_alloc_type.dims(),
                                             new_in_alloc_type.data_layout(),
                                             new_in_alloc_type.lod(),
                                             new_in_alloc_type.offset());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else if (new_in_type.isa<AllocatedDenseTensorArrayType>()) {
        // allocated type
        auto in_place =
            new_in_type.dyn_cast<AllocatedDenseTensorArrayType>().place();

        // get input args def type
        auto args_def = kernel.args_def();
        auto input_defs = args_def.input_defs();

        auto input_backend = ChooseInputBackend(
            kernel, tensor_param_index, kernel_key.backend());
        auto dst_backend = DeriveBackend(
            op_item->name(), place, op_info_parser, input_backend, i);
        VLOG(6) << "Infer kernel backend from input " << i << " of op ";
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
              new_in_type.dyn_cast<AllocatedDenseTensorArrayType>();
          auto out_type = AllocatedDenseTensorArrayType::get(
              ctx,
              out_place,
              new_in_alloc_type.dtype(),
              new_in_alloc_type.data_layout());
          new_in = AddPlaceTransferOp(
              new_in, out_type, in_place, out_place, kernel_key, block);
        }
      } else {
        PADDLE_THROW(
            phi::errors::Unimplemented("only support allocated dense tensor "
                                       "type and selected rows for now"));
      }
    }

    // 2. dtype transfer
    if (op_info_parser != nullptr) {
      std::string var_name = op_info_parser->InputNames()[i];
      auto fake_tensors = PrepareFakeTensors(new_in);
      if (!fake_tensors.empty()) {
        const phi::KernelKey expected_kernel_key = kernel_key;
        const phi::DataType kernel_dtype_for_var =
            GetKernelTypeforVar(op_item,
                                var_name,
                                (*fake_tensors[0]).dtype(),
                                &expected_kernel_key);

        bool check_dtype_transfer = NeedTransformDataType(
            expected_kernel_key.dtype(), kernel_dtype_for_var);
        if (check_dtype_transfer) {
          VLOG(4) << "trans input: " << var_name << "'s dtype from "
                  << kernel_dtype_for_var << " to "
                  << expected_kernel_key.dtype();

          auto out_place = phi::TransToPhiPlace(expected_kernel_key.backend());
          new_in = AddDtypeTransferOp(new_in,
                                      block,
                                      kernel_key,
                                      place,
                                      out_place,
                                      kernel_dtype_for_var,
                                      expected_kernel_key.dtype());
        }
      }
    }
    vec_inputs.push_back(new_in);
  }
  return vec_inputs;
}

void AddShadowFeedOpForDataOrFeed(
    const phi::Place& place,
    pir::Operation* op_item,
    pir::Operation* kernel_op,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  bool feed_op_add_shadow_feed =
      (op_item->isa<FeedOp>()) && platform::is_gpu_place(place);
  bool data_op_add_shadow_feed =
      (op_item->isa<DataOp>()) && platform::is_gpu_place(place) &&
      (kernel_op->attributes()
           .at("place")
           .dyn_cast<PlaceAttribute>()
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
        {"kernel_key", KernelAttribute::get(ctx, shadow_key)}};

    auto out_type = AllocatedDenseTensorType::get(
        ctx,
        phi::TransToPhiPlace(shadow_key.backend()),
        op_item->result(0).type().dyn_cast<DenseTensorType>());

    pir::OpInfo phi_kernel_op_info =
        ctx->GetRegisteredOpInfo(PhiKernelOp::name());
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

pir::Operation* BuildKernelOp(
    const std::string& kernel_fn_str,
    const phi::KernelKey& kernel_key,
    const std::vector<pir::Value>& vec_inputs,
    const std::vector<pir::Type>& op_output_types,
    pir::Operation* op_item,
    pir::Block* block,
    pir::IrContext* ctx,
    std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair,
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  std::unordered_map<std::string, pir::Attribute> op_attribute{
      {"op_name", pir::StrAttribute::get(ctx, op_item->name())},
      {"kernel_name", pir::StrAttribute::get(ctx, kernel_fn_str)},
      {"kernel_key", KernelAttribute::get(ctx, kernel_key)}};
  auto op_attr_map = op_item->attributes();

  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }

  if (op_item->HasTrait<InplaceTrait>()) {
    op_attribute.emplace("is_inplace", pir::BoolAttribute::get(ctx, true));
  }

  pir::OpInfo phi_kernel_op_info =
      ctx->GetRegisteredOpInfo(PhiKernelOp::name());

  pir::OpInfo legacy_kernel_op_info =
      ctx->GetRegisteredOpInfo(LegacyKernelOp::name());
  pir::Operation* op = nullptr;
  if (IsLegacyOp(op_item->name())) {
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
    std::unordered_map<pir::Value, pir::Value>* map_value_pair) {
  auto inputs_by_data_op = GetInputsByDataOp(block);

  for (auto& op_item : *block) {
    VLOG(6) << "op name " << op_item.name();
    if ((op_item.isa<FeedOp>()) &&
        inputs_by_data_op.count(op_item.attributes()
                                    .at("name")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString())) {
      VLOG(6) << "Skip FeedOp while lowering to kernel pass";
      continue;
    }

    // HandleSpecialOp
    if (SpecialLowerOps.count(op_item.name())) {
      VLOG(6) << "Handle Special Op: [" << op_item.name()
              << "] while lowering to kernel pass";
      HandleForSpecialOp(
          place, &op_item, new_block, ctx, map_op_pair, map_value_pair);
      continue;
    }

    auto op_info_parser = GetOpYamlInfoParser(&op_item);
    auto kernel_name = GetKernelName(op_info_parser.get(), &op_item);
    auto kernel_key = GetKernelKey(
        &op_item, place, kernel_name, *map_value_pair, op_info_parser.get());
    VLOG(6) << "kernel type " << kernel_key;

    // build output type
    auto op_output_types = BuildOutputs(&op_item, kernel_name, kernel_key, ctx);
    // build input
    auto vec_inputs = BuildInputs(&op_item,
                                  kernel_name,
                                  kernel_key,
                                  place,
                                  op_info_parser.get(),
                                  ctx,
                                  map_op_pair,
                                  map_value_pair,
                                  new_block);

    // build op
    pir::Operation* op = BuildKernelOp(kernel_name,
                                       kernel_key,
                                       vec_inputs,
                                       op_output_types,
                                       &op_item,
                                       new_block,
                                       ctx,
                                       map_op_pair,
                                       map_value_pair);

    AddShadowFeedOpForDataOrFeed(
        place, &op_item, op, new_block, ctx, map_op_pair, map_value_pair);
  }
}

std::unique_ptr<pir::Program> PdOpLowerToKernelPass(pir::Program* prog,
                                                    phi::Place place) {
  if (FLAGS_print_ir) {
    std::cout << "IR before lowering = " << *prog << std::endl;
  }

  auto program = std::make_unique<pir::Program>(pir::IrContext::Instance());

  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();

  std::unordered_map<pir::Operation*, pir::Operation*> map_op_pair;
  std::unordered_map<pir::Value, pir::Value> map_value_pair;

  ProcessBlock(
      place, block, program->block(), ctx, &map_op_pair, &map_value_pair);

  if (FLAGS_print_ir) {
    std::cout << "IR after lowering = " << *program << std::endl;
  }

  return program;
}
}  // namespace dialect
}  // namespace paddle
