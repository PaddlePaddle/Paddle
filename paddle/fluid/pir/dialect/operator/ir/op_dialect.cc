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

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/transforms/param_to_variable.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/interface_value.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/manual_onednn_op.h"
#endif

namespace paddle {
namespace dialect {

struct CombineOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    const auto shape_data_list = [&] {
      symbol::TensorListShapeOrDataDimExprs shape_data_list;
      for (size_t i = 0; i < op->num_operands(); ++i) {
        IR_ENFORCE(op->operand(i).type().dyn_cast<DenseTensorType>(),
                   "Currently InferSymbolicShape of CombineOp only support "
                   "DenseTensorType.");

        shape_data_list.emplace_back(
            shape_analysis->GetShapeOrDataForValue(op->operand_source(i))
                .dyn_cast<symbol::TensorShapeOrDataDimExprs>());
      }
      return shape_data_list;
    }();

    symbol::ShapeOrDataDimExprs shape_data{shape_data_list};
    shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;
  }

  CombineOpInferSymbolicShapeInterfaceModel()
      : InferSymbolicShapeInterface::Concept(InferSymbolicShape) {}
};

struct ConstantOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    IR_ENFORCE(op->result(0).type().dyn_cast<DenseTensorType>(),
               "Currently InferSymbolicShape of ConstantOp only support "
               "DenseTensorType result.");

    const std::vector<symbol::DimExpr> out_dims = [op] {
      std::vector<symbol::DimExpr> dims;
      const std::vector<int64_t> result_dims = common::vectorize(
          op->result(0).type().dyn_cast<pir::DenseTensorType>().dims());
      for (size_t i = 0; i < result_dims.size(); i++) {
        dims.emplace_back(result_dims[i]);
      }
      return dims;
    }();

    shape_analysis->SetShapeOrDataForValue(
        op->result(0),
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(out_dims)});

    return true;
  }

  ConstantOpInferSymbolicShapeInterfaceModel()
      : InferSymbolicShapeInterface::Concept(InferSymbolicShape) {}
};

struct ParameterOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    pir::Value res0 = op->result(0);

    std::vector<int64_t> dims =
        common::vectorize(res0.type().dyn_cast<pir::DenseTensorType>().dims());

    // TODO(zhangbopd): check whether it's right for other cases
    std::vector<symbol::DimExpr> sym_shape;
    for (int64_t dim : dims) {
      symbol::DimExpr dim_expr;
      if (dim == -1) {
        symbol::DimExpr res_dim_expr(shape_analysis->GetNextSymName());
        dim_expr = res_dim_expr;
      } else {
        symbol::DimExpr res_dim_expr(dim);
        dim_expr = res_dim_expr;
      }
      sym_shape.push_back(dim_expr);
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_shape)};

    shape_analysis->SetShapeOrDataForValue(res0, shape_data);

    return true;
  }

  ParameterOpInferSymbolicShapeInterfaceModel()
      : InferSymbolicShapeInterface::Concept(InferSymbolicShape) {}
};

struct ShadowOutputOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    pir::Value operand_source = op->operand_source(0);
    auto input_shapeordata =
        shape_analysis->GetShapeOrDataForValue(operand_source);

    symbol::ShapeOrDataDimExprs shape_data = input_shapeordata;
    pir::shape::SetShapeAttrForOp(op, shape_data);

    return true;
  }

  ShadowOutputOpInferSymbolicShapeInterfaceModel()
      : InferSymbolicShapeInterface::Concept(InferSymbolicShape) {}
};

struct YieldOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    // Since YieldOp has no output, just return true
    return true;
  }

  YieldOpInferSymbolicShapeInterfaceModel()
      : InferSymbolicShapeInterface::Concept(InferSymbolicShape) {}
};

OperatorDialect::OperatorDialect(pir::IrContext* ctx)
    : pir::Dialect(name(), ctx, pir::TypeId::get<OperatorDialect>()) {
  initialize();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  auto info = ctx->GetRegisteredOpInfo(pir::TuplePushOp::name());
  info.AttachInterface(std::move(
      pir::InterfaceValue::Get<VjpInterface, TuplePushOpVjpInterfaceModel>()));

  info = ctx->GetRegisteredOpInfo(pir::CombineOp::name());
  info.AttachInterface(std::move(
      pir::InterfaceValue::Get<InferSymbolicShapeInterface,
                               CombineOpInferSymbolicShapeInterfaceModel>()));

  info = ctx->GetRegisteredOpInfo(pir::ParameterOp::name());
  info.AttachInterface(std::move(
      pir::InterfaceValue::Get<InferSymbolicShapeInterface,
                               ParameterOpInferSymbolicShapeInterfaceModel>()));

  info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
  info.AttachInterface(
      std::move(pir::InterfaceValue::Get<
                InferSymbolicShapeInterface,
                ShadowOutputOpInferSymbolicShapeInterfaceModel>()));

  info = ctx->GetRegisteredOpInfo(pir::YieldOp::name());
  info.AttachInterface(std::move(
      pir::InterfaceValue::Get<InferSymbolicShapeInterface,
                               YieldOpInferSymbolicShapeInterfaceModel>()));
}

void PrintTypeImpl(pir::Type type, std::ostream& os) {
  os << type.dialect().name();
  os << '.';
  if (auto selected_rows_type = type.dyn_cast<SelectedRowsType>()) {
    os << "selectedrows<";
    for (auto d : common::vectorize(selected_rows_type.dims())) {
      os << d;
      os << "x";
    }
    selected_rows_type.dtype().Print(os);
    os << ">";
  } else if (auto tensor_array_type = type.dyn_cast<DenseTensorArrayType>()) {
    os << "tensor_array<";
    tensor_array_type.dtype().Print(os);
    os << ">";
  }
}
void PrintAttributeImpl(pir::Attribute attr, std::ostream& os) {
  os << "(" << attr.dialect().name();
  os << '.';
  if (auto int_array_attr = attr.dyn_cast<IntArrayAttribute>()) {
    phi::IntArray data = int_array_attr.data();
    os << "IntArray)"
       << "[";
    const auto& inner_data = data.GetData();
    pir::detail::PrintInterleave(
        inner_data.begin(),
        inner_data.end(),
        [&os](int64_t i) { os << i; },
        [&os]() { os << ","; });
    os << "]";
  } else if (auto data_type_attr = attr.dyn_cast<DataTypeAttribute>()) {
    os << "DataType)" << data_type_attr.data();
  } else if (auto place_type_attr = attr.dyn_cast<PlaceAttribute>()) {
    os << "Place)" << place_type_attr.data();
  } else if (auto data_layout_attr = attr.dyn_cast<DataLayoutAttribute>()) {
    os << "DataLayout)" << data_layout_attr.data();
  } else {
    os << "<#AttrNotImplemented>";
  }
}

void PrintOperationImpl(pir::Operation* op,
                        pir::IrPrinter& printer) {  // NOLINT
  if (auto if_op = op->dyn_cast<IfOp>()) {
    if_op.Print(printer);
  } else if (auto while_op = op->dyn_cast<WhileOp>()) {
    while_op.Print(printer);
  } else if (auto pylayer_op = op->dyn_cast<PyLayerOp>()) {
    pylayer_op.Print(printer);
  } else {
    printer.PrintGeneralOperation(op);
  }
}

void OperatorDialect::initialize() {
  RegisterTypes<paddle::dialect::SelectedRowsType,
                paddle::dialect::DenseTensorArrayType>();

  RegisterAttributes<paddle::dialect::IntArrayAttribute,
                     paddle::dialect::DataTypeAttribute,
                     paddle::dialect::PlaceAttribute,
                     paddle::dialect::DataLayoutAttribute>();

  // NOTE(zhangbo9674): GET_OP_LIST is defined in pd_op.h which is
  // generated by op_gen.py, see details in
  // paddle/fluid/pir/dialect/CMakeLists.txt.
  // NOTE(Ruting)GET_MANUAL_OP_LIST is define in manual_op.h"
  // use RegisterOps when list has more than two ops.

  // NOTE(cocoshe): VS2017 has a limit on the length of template
  // parameters, which causes "fatal error C1202".
  // Split GET_OP_LIST into two part on WIN32 here.
#ifdef WIN32
  RegisterOps<
#define GET_OP_LIST1
#include "paddle/fluid/pir/dialect/operator/ir/pd_op_info.cc"  // NOLINT
      >();

  RegisterOps<
#define GET_OP_LIST2
#include "paddle/fluid/pir/dialect/operator/ir/pd_op_info.cc"  // NOLINT
      >();
#else
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/pd_op_info.cc"  // NOLINT
      >();
#endif
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc"  // NOLINT
      >();

  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.cc"  // NOLINT
      >();

#ifdef PADDLE_WITH_DNNL
  RegisterOps<
#define GET_OP_LIST
#include "paddle/fluid/pir/dialect/operator/ir/manual_onednn_op.cc"  // NOLINT
      >();
#endif
  RegisterInterfaces<ParameterConvertInterface>();
}

void OperatorDialect::PrintType(pir::Type type, std::ostream& os) const {
  PrintTypeImpl(type, os);
}

void OperatorDialect::PrintAttribute(pir::Attribute attr,
                                     std::ostream& os) const {
  PrintAttributeImpl(attr, os);
}

pir::Attribute OperatorDialect::ParseAttribute(
    pir::IrParser& parser) {  // NOLINT
  std::string type_name = parser.ConsumeToken().val_;
  std::string attribute_name =
      type_name.substr(type_name.find('.') + 1, std::string::npos);
  parser.ConsumeAToken(")");
  if (attribute_name == "IntArray") {
    return IntArrayAttribute::Parse(parser);
  } else if (attribute_name == "DataType") {
    return DataTypeAttribute::Parse(parser);
  } else if (attribute_name == "Place") {
    return PlaceAttribute::Parse(parser);
  } else if (attribute_name == "DataLayout") {
    return DataLayoutAttribute::Parse(parser);
  } else {
    IR_THROW("No function to parse " + attribute_name + " exists!" +
             parser.GetErrorLocationInfo());
  }
}

pir::OpPrintFn OperatorDialect::PrintOperation(pir::Operation* op) const {
  if (op->isa<IfOp>() || op->isa<WhileOp>()) {
    return PrintOperationImpl;
  }
  return nullptr;
}

class IdManager {
 public:
  static IdManager& Instance() {
    static IdManager instance;
    return instance;
  }

  ~IdManager() {
    for (auto id : ids_) {
      delete id;
    }
    ids_.clear();
  }

  pir::TypeId CreateId() {
    pir::detail::UniqueingId* unique_id = new pir::detail::UniqueingId();
    ids_.push_back(unique_id);
    return ids_.back()->id();
  }

 private:
  std::vector<pir::detail::UniqueingId*> ids_;
};

class AttributeManager {
 public:
  static AttributeManager& Instance() {
    static AttributeManager instance;
    return instance;
  }

  ~AttributeManager() {
    for (size_t i = 0; i < char_pointers_.size(); i++) {
      for (size_t j = 0; j < pointers_size_[i]; j++) {
        delete char_pointers_[i][j];
      }
      delete char_pointers_[i];
    }
    char_pointers_.clear();
    pointers_size_.clear();
  }

  const char** ToCharPointers(const std::vector<std::string>& attr_names) {
    const char** char_pointers = new const char*[attr_names.size()];
    for (size_t i = 0; i < attr_names.size(); i++) {
      const std::string& attr_name = attr_names[i];
      char* ptr = new char[attr_name.size() + 1];
      snprintf(ptr, attr_name.size() + 1, "%s", attr_name.c_str());
      char_pointers[i] = ptr;
    }
    pointers_size_.push_back(attr_names.size());
    char_pointers_.push_back(char_pointers);
    return char_pointers;
  }

 private:
  std::vector<const char**> char_pointers_;
  std::vector<uint32_t> pointers_size_;
};

struct CustomOpInfoInterfaceModel : public OpYamlInfoInterface::Concept {
  static OpInfoTuple GetPirOpInfo(const std::string& pir_op_name) {
    const auto& op_meta =
        paddle::framework::detail::GetOpInfoByPirName(pir_op_name);
    const auto* grad_op_meta_ptr =
        paddle::framework::detail::GetGradOpInfoByFwdPirName(pir_op_name);
    std::vector<paddle::dialect::OpInputInfo> inputs_info;
    std::vector<paddle::dialect::OpAttributeInfo> attributes_info;
    std::vector<paddle::dialect::OpOutputInfo> outputs_info;
    std::vector<std::string> param_names;
    // translate input info
    auto& op_input_names = OpMetaInfoHelper::GetInputs(op_meta);
    for (const auto& input_name : op_input_names) {
      param_names.push_back(input_name);
      bool is_optional = false;
      bool with_grad_semantic = false;
      std::string input_type = "paddle::dialect::DenseTensorType";
      if (paddle::framework::detail::IsOptionalVar(input_name)) {
        is_optional = true;
      }
      if (paddle::framework::detail::IsDuplicableVar(input_name)) {
        input_type = "pir::VectorType<paddle::dialect::DenseTensorType>";
      }
      if (grad_op_meta_ptr) {
        const auto& grad_op_name =
            OpMetaInfoHelper::GetOpName(*grad_op_meta_ptr);
        auto& grad_op_output_names =
            OpMetaInfoHelper::GetOutputs(*grad_op_meta_ptr);
        bool is_double_grad_op =
            (grad_op_name.find("_grad_grad") != grad_op_name.npos) ? true
                                                                   : false;
        for (auto& grad_op_output_name : grad_op_output_names) {
          auto fwd_input_name = paddle::framework::detail::NoGrad(
              grad_op_output_name, is_double_grad_op);
          if (input_name == fwd_input_name) {
            with_grad_semantic = true;
            break;
          }
        }
      }
      // Now, we only support dense tensor as input.
      inputs_info.push_back(paddle::dialect::OpInputInfo{input_name,
                                                         input_type,
                                                         is_optional,
                                                         false,
                                                         false,
                                                         with_grad_semantic});
    }

    // translate attr info
    auto& op_attrs = OpMetaInfoHelper::GetAttrs(op_meta);
    for (const auto& op_attr : op_attrs) {
      auto attr_name_and_type = paddle::ParseAttrStr(op_attr);
      auto attr_name = attr_name_and_type[0];
      auto attr_type_str = attr_name_and_type[1];
      param_names.push_back(attr_name);
      if (AttrTypeMap().find(attr_type_str) == AttrTypeMap().end()) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, "
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_type_str));
      }
      std::string attr_pir_type = AttrTypeMap().at(attr_type_str);
      attributes_info.emplace_back(attr_name, attr_pir_type, "");
    }

    // translate output info
    auto& op_output_names = OpMetaInfoHelper::GetOutputs(op_meta);
    for (const auto& output_name : op_output_names) {
      bool is_optional = false;
      if (paddle::framework::detail::IsOptionalVar(output_name)) {
        is_optional = true;
      }
      // Now, we only support dense tensor as output.
      outputs_info.push_back(paddle::dialect::OpOutputInfo{
          output_name, "paddle::dialect::DenseTensorType", is_optional, false});
    }

    auto& inplace_maps = OpMetaInfoHelper::GetInplaceReverseMap(op_meta);

    if (!inplace_maps.empty()) {
      VLOG(3) << "Register Custom Operator: op inplace_map: "
              << string::join_strings(inplace_maps, ',', [](auto& pair) {
                   return pair.first + ": " + pair.second;
                 });
    }

    std::vector<std::pair<std::string, std::string>> vec_inplace;
    for (const auto& inplace_map : inplace_maps) {
      vec_inplace.emplace_back(inplace_map);
    }

    // we only need kernel params name in run_time_info
    paddle::dialect::OpRunTimeInfo run_time_info =
        paddle::dialect::OpRunTimeInfo(
            "", {}, "", param_names, {}, {}, vec_inplace, {});

    return std::make_tuple(
        inputs_info, attributes_info, outputs_info, run_time_info, "");
  }

  CustomOpInfoInterfaceModel() : OpYamlInfoInterface::Concept(GetPirOpInfo) {}
};

struct CustomOpVjpInterfaceModel : public VjpInterface::Concept {
  static std::vector<std::vector<pir::Value>> CustomOpVjp(
      pir::Operation* op,
      const std::vector<std::vector<pir::Value>>& inputs_,
      const std::vector<std::vector<pir::Value>>& outputs,
      const std::vector<std::vector<pir::Value>>& out_grads,
      const std::vector<std::vector<bool>>& stop_gradients) {
    std::string pir_op_name = op->name();
    const auto& fwd_op_meta_info =
        paddle::framework::detail::GetOpInfoByPirName(pir_op_name);
    const auto& fwd_inputs_name =
        paddle::OpMetaInfoHelper::GetInputs(fwd_op_meta_info);
    const auto& fwd_attrs_name =
        paddle::OpMetaInfoHelper::GetAttrs(fwd_op_meta_info);
    const auto& fwd_outputs_name =
        paddle::OpMetaInfoHelper::GetOutputs(fwd_op_meta_info);

    const auto* bwd_op_meta_info_ptr =
        paddle::framework::detail::GetGradOpInfoByFwdPirName(pir_op_name);
    if (bwd_op_meta_info_ptr == nullptr) {
      PADDLE_THROW("Custom op : " + pir_op_name + " doesn't support its grad.");
    }
    const auto& bwd_op_meta_info = *bwd_op_meta_info_ptr;
    const auto& bwd_inputs_name =
        paddle::OpMetaInfoHelper::GetInputs(bwd_op_meta_info);
    const auto& bwd_outputs_name =
        paddle::OpMetaInfoHelper::GetOutputs(bwd_op_meta_info);
    const auto& bwd_inplace_map =
        paddle::OpMetaInfoHelper::GetInplaceMap(bwd_op_meta_info);
    const auto& bwd_op_name =
        paddle::OpMetaInfoHelper::GetOpName(bwd_op_meta_info);
    std::string bwd_pir_op_name =
        paddle::framework::kCustomDialectPrefix + bwd_op_name;
    if (!bwd_inplace_map.empty()) {
      // inplace case
      bwd_pir_op_name += "_";
    }
    auto infershape_func = OpMetaInfoHelper::GetInferShapeFn(bwd_op_meta_info);
    auto inferdtype_func = OpMetaInfoHelper::GetInferDtypeFn(bwd_op_meta_info);
    PADDLE_ENFORCE_EQ(
        inputs_.size(),
        fwd_inputs_name.size(),
        paddle::platform::errors::InvalidArgument(
            "Custom op: %s inputs size should be %d, but now is %d.",
            pir_op_name,
            fwd_inputs_name.size(),
            inputs_.size()));
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        fwd_outputs_name.size(),
        paddle::platform::errors::InvalidArgument(
            "Custom op: %s outputs size should be %d, but now is %d.",
            pir_op_name,
            fwd_outputs_name.size(),
            outputs.size()));

    PADDLE_ENFORCE_EQ(
        out_grads.size(),
        fwd_outputs_name.size(),
        paddle::platform::errors::InvalidArgument(
            "Custom op: %s outputs grad size should be %d, but now is %d.",
            pir_op_name,
            fwd_outputs_name.size(),
            out_grads.size()));

    bool is_double_grad_op =
        (bwd_pir_op_name.find("_grad_grad") != pir_op_name.npos) ? true : false;
    pir::IrContext* ctx = pir::IrContext::Instance();
    pir::OpInfo pir_info = ctx->GetRegisteredOpInfo(bwd_pir_op_name);
    pir::OperationArgument argument(pir_info);
    std::vector<pir::Value> argument_inputs;
    std::vector<pir::Type> argument_outputs;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<DataType> input_dtypes;
    std::unordered_map<std::string, int> input_name2id_map;
    std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;
    std::vector<std::vector<DataType>> vec_input_dtypes;
    std::unordered_map<std::string, int> vec_input_name2id_map;
    std::vector<paddle::any> custom_attrs;
    auto GetInputLocation =
        [&](const std::string& grad_op_input_name) -> std::pair<int, int> {
      auto fwd_inputs_name_iter = std::find(
          fwd_inputs_name.begin(), fwd_inputs_name.end(), grad_op_input_name);
      auto fwd_outputs_name_iter = std::find(
          fwd_outputs_name.begin(), fwd_outputs_name.end(), grad_op_input_name);
      bool is_grad_var = paddle::framework::detail::IsGradVar(
          grad_op_input_name, is_double_grad_op);
      if (fwd_inputs_name_iter != fwd_inputs_name.end()) {
        int index =
            std::distance(fwd_inputs_name.begin(), fwd_inputs_name_iter);
        return std::make_pair(0, index);
      } else if (fwd_outputs_name_iter != fwd_outputs_name.end()) {
        int index =
            std::distance(fwd_outputs_name.begin(), fwd_outputs_name_iter);
        return std::make_pair(1, index);
      } else if (is_grad_var) {
        auto fwd_output_name = paddle::framework::detail::NoGrad(
            grad_op_input_name, is_double_grad_op);
        fwd_outputs_name_iter = std::find(
            fwd_outputs_name.begin(), fwd_outputs_name.end(), fwd_output_name);
        if (fwd_outputs_name_iter != fwd_outputs_name.end()) {
          int index =
              std::distance(fwd_outputs_name.begin(), fwd_outputs_name_iter);
          return std::make_pair(2, index);
        } else {
          PADDLE_THROW(paddle::platform::errors::NotFound(
              "Can't find the grad op input:%s, please check your register "
              "grad op whether has correct input name",
              grad_op_input_name));
        }
      } else {
        PADDLE_THROW(paddle::platform::errors::NotFound(
            "Can't find the grad op input:%s, please check your register grad "
            "op whether has correct input name",
            grad_op_input_name));
      }
    };

    // Construct custom grad op inputs
    int input_index = 0;
    int vec_input_index = 0;
    for (size_t i = 0; i < bwd_inputs_name.size(); ++i) {
      const auto& bwd_input_name = bwd_inputs_name.at(i);
      const auto input_location = GetInputLocation(bwd_input_name);
      std::vector<pir::Value> input_values;
      if (input_location.first == 0) {
        // grad op input is in inputs_
        input_values = inputs_[input_location.second];
      } else if (input_location.first == 1) {
        // grad op input is in outputs
        input_values = outputs[input_location.second];
      } else {
        // grad op input is in out_grads
        input_values = out_grads[input_location.second];
      }

      if (input_values.size() > 1) {
        std::vector<std::vector<int64_t>> tmp_input_shapes;
        std::vector<phi::DataType> tmp_input_dtypes;
        vec_input_name2id_map[bwd_input_name] = vec_input_index;
        vec_input_index++;
        for (auto& input_value : input_values) {
          paddle::dialect::DenseTensorType input_tensor =
              input_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
          tmp_input_shapes.push_back(phi::vectorize(input_tensor.dims()));
          tmp_input_dtypes.push_back(
              paddle::dialect::TransToPhiDataType(input_tensor.dtype()));
        }
        vec_input_shapes.push_back(tmp_input_shapes);
        vec_input_dtypes.push_back(tmp_input_dtypes);
        auto input_value = paddle::dialect::builtin_combine(input_values);
        argument_inputs.push_back(input_value);
      } else {
        input_name2id_map[bwd_input_name] = input_index;
        input_index++;
        pir::Value input_value = input_values[0];  // NOLINT
        paddle::dialect::DenseTensorType input_tensor =
            input_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
        input_shapes.push_back(phi::vectorize(input_tensor.dims()));
        input_dtypes.push_back(
            paddle::dialect::TransToPhiDataType(input_tensor.dtype()));
        argument_inputs.push_back(input_value);
      }
    }
    argument.AddInputs(argument_inputs);
    // Construct custom grad op attr
    for (size_t i = 0; i < fwd_attrs_name.size(); ++i) {
      const auto& fwd_attr = fwd_attrs_name.at(i);
      std::vector<std::string> attr_name_and_type =
          paddle::ParseAttrStr(fwd_attr);
      auto fwd_attr_name = attr_name_and_type[0];
      auto fwd_op_attr = op->attribute(fwd_attr_name);
      custom_attrs.push_back(paddle::dialect::TransAttrToAny(fwd_op_attr));
      argument.AddAttribute(fwd_attr_name, fwd_op_attr);
    }

    // Run Compile InferMeta
    std::vector<std::vector<int64_t>> output_shapes =
        paddle::framework::RunInferShape(infershape_func,
                                         bwd_op_meta_info,
                                         input_shapes,
                                         input_name2id_map,
                                         vec_input_shapes,
                                         vec_input_name2id_map,
                                         custom_attrs);
    std::vector<phi::DataType> output_dtypes =
        paddle::framework::RunInferDtype(inferdtype_func,
                                         bwd_op_meta_info,
                                         input_dtypes,
                                         input_name2id_map,
                                         vec_input_dtypes,
                                         vec_input_name2id_map,
                                         custom_attrs);
    size_t all_values_num = 0;
    // output name -> value num (that output should hold)
    std::unordered_map<std::string, size_t> output_name2value_num;
    for (size_t i = 0; i < bwd_outputs_name.size(); ++i) {
      const auto& bwd_output_name = bwd_outputs_name.at(i);
      if (paddle::framework::detail::IsDuplicableVar(bwd_output_name)) {
        const auto& bwd_input = paddle::framework::detail::NoGrad(
            bwd_output_name, is_double_grad_op);
        auto index = vec_input_name2id_map[bwd_input];
        auto& input_shapes = vec_input_shapes[index];
        output_name2value_num[bwd_output_name] = input_shapes.size();
        all_values_num += input_shapes.size();
      } else {
        output_name2value_num[bwd_output_name] = 1;
        all_values_num++;
      }
    }
    PADDLE_ENFORCE_EQ(
        output_shapes.size(),
        all_values_num,
        phi::errors::InvalidArgument(
            "The number of output shapes after running custom operator's "
            "InferShapeFunc is wrong, "
            "expected contains %d Tensors' shape, but actually contains %d "
            "Tensors' shape",
            all_values_num,
            output_shapes.size()));

    PADDLE_ENFORCE_EQ(
        output_dtypes.size(),
        all_values_num,
        phi::errors::InvalidArgument(
            "The number of output dtypes after running custom operator's "
            "InferDtypeFunc is wrong, "
            "expected contains %d Tensors' dtype, but actually contains %d "
            "Tensors' dtype",
            all_values_num,
            output_dtypes.size()));

    // Construct custom grad op outputs
    size_t value_index = 0;
    for (size_t i = 0; i < bwd_outputs_name.size(); ++i) {
      const auto& bwd_output_name = bwd_outputs_name.at(i);
      if (paddle::framework::detail::IsDuplicableVar(bwd_output_name)) {
        auto value_num = output_name2value_num[bwd_output_name];
        std::vector<pir::Type> out_types;
        for (size_t j = 0; j < value_num; ++j) {
          auto ddims = phi::make_ddim(output_shapes[value_index]);
          auto dtype = output_dtypes[value_index];
          phi::DataLayout layout{DataLayout::NCHW};
          phi::LoD lod;
          out_types.push_back(paddle::dialect::DenseTensorType::get(
              pir::IrContext::Instance(),
              paddle::dialect::TransToIrDataType(dtype),
              ddims,
              layout,
              lod,
              0));
          value_index++;
        }
        pir::Type out_vector_type =
            pir::VectorType::get(pir::IrContext::Instance(), out_types);
        argument_outputs.push_back(out_vector_type);
      } else {
        auto ddims = phi::make_ddim(output_shapes[value_index]);
        auto dtype = output_dtypes[value_index];
        phi::DataLayout layout{DataLayout::NCHW};
        phi::LoD lod;
        auto out_type = paddle::dialect::DenseTensorType::get(
            pir::IrContext::Instance(),
            paddle::dialect::TransToIrDataType(dtype),
            ddims,
            layout,
            lod,
            0);
        argument_outputs.push_back(out_type);
        value_index++;
      }
    }
    argument.AddOutputs(argument_outputs.begin(), argument_outputs.end());
    // Build Operation
    std::vector<pir::Value> op_results;
    pir::Operation* bwd_op =
        paddle::dialect::ApiBuilder::Instance().GetBuilder()->Build(
            std::move(argument));

    // Init result
    std::vector<std::vector<pir::Value>> res;
    res.resize(stop_gradients.size());
    for (size_t i = 0; i < stop_gradients.size(); ++i) {
      res[i].resize(stop_gradients[i].size());
    }
    // Build result and apply stop gradients
    for (size_t i = 0; i < bwd_outputs_name.size(); ++i) {
      const auto& bwd_output_name = bwd_outputs_name.at(i);
      const auto& fwd_input =
          paddle::framework::detail::NoGrad(bwd_output_name, is_double_grad_op);
      auto fwd_inputs_name_iter =
          std::find(fwd_inputs_name.begin(), fwd_inputs_name.end(), fwd_input);
      if (paddle::framework::detail::IsDuplicableVar(bwd_output_name)) {
        PADDLE_ENFORCE_NE(
            fwd_inputs_name_iter,
            fwd_inputs_name.end(),
            paddle::platform::errors::InvalidArgument(
                "Custom op: %s output %s is a Vec output. It should have the "
                "forward input that need calculate gradients.",
                pir_op_name,
                bwd_output_name));
        int index =
            std::distance(fwd_inputs_name.begin(), fwd_inputs_name_iter);
        auto split_op =
            ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
                bwd_op->result(i));
        res[index] = split_op.outputs();
      } else {
        if (fwd_inputs_name_iter != fwd_inputs_name.end()) {
          int index =
              std::distance(fwd_inputs_name.begin(), fwd_inputs_name_iter);
          res[index][0] = bwd_op->result(i);
        } else {
          // Situation that has only one input and only one output. If not meet
          // this condition, it will throw error when run infer shape.
          res[0][0] = bwd_op->result(0);
        }
      }
    }
    return res;
  }

  CustomOpVjpInterfaceModel() : VjpInterface::Concept(CustomOpVjp) {}
};

CustomOpDialect::CustomOpDialect(pir::IrContext* context)
    : pir::Dialect(name(), context, pir::TypeId::get<CustomOpDialect>()) {}

void CustomOpDialect::PrintType(pir::Type type, std::ostream& os) const {
  PrintTypeImpl(type, os);
}

void CustomOpDialect::PrintAttribute(pir::Attribute attr,
                                     std::ostream& os) const {
  PrintAttributeImpl(attr, os);
}

pir::OpPrintFn CustomOpDialect::PrintOperation(pir::Operation* op) const {
  return nullptr;
}

void CustomOpDialect::RegisterCustomOp(const paddle::OpMetaInfo& op_meta) {
  pir::TypeId id = IdManager::Instance().CreateId();
  std::string op_name = paddle::framework::kCustomDialectPrefix +
                        OpMetaInfoHelper::GetOpName(op_meta);
  std::vector<pir::TypeId> traits;

  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(op_meta);
  if (!inplace_map.empty()) {
    op_name += "_";
    traits.push_back(pir::TypeId::get<paddle::dialect::InplaceTrait>());
  }

  char* op_name_c = new char[op_name.size() + 1];
  snprintf(op_name_c, op_name.size() + 1, "%s", op_name.c_str());
  op_names_.push_back(op_name_c);

  auto& op_attrs = OpMetaInfoHelper::GetAttrs(op_meta);
  std::vector<std::string> attr_names;
  for (const auto& op_attr : op_attrs) {
    auto attr_name_and_type = paddle::ParseAttrStr(op_attr);
    auto attr_name = attr_name_and_type[0];
    attr_names.push_back(attr_name);
  }
  const char** attr_name =
      AttributeManager::Instance().ToCharPointers(attr_names);
  uint32_t attr_num = attr_names.size();

  std::set<pir::InterfaceValue> interface_values;
  pir::InterfaceValue op_info_interface =
      pir::InterfaceValue::Get<OpYamlInfoInterface,
                               CustomOpInfoInterfaceModel>();
  interface_values.insert(std::move(op_info_interface));

  if (paddle::framework::detail::HasGradOp(op_name)) {
    pir::InterfaceValue vjp_interface =
        pir::InterfaceValue::Get<VjpInterface, CustomOpVjpInterfaceModel>();
    interface_values.insert(std::move(vjp_interface));
  }

  // Currently we set empty verify function and will reset it if it is used in
  // future.
  pir::VerifyPtr verify_func = [](pir::Operation* op) {};
  ir_context()->RegisterOpInfo(this,
                               id,
                               op_names_.back(),
                               std::move(interface_values),
                               traits,
                               attr_num,
                               attr_name,
                               verify_func,
                               verify_func);
}
}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OperatorDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CustomOpDialect)
