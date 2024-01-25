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
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/transforms/param_to_variable.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/interface_value.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/dialect/shape/ir/shape_attribute.h"

namespace paddle {
namespace dialect {

static std::unordered_map<std::string, std::string> kCustomTypeMap = {
    {"bool", "pir::BoolAttribute"},
    {"int", "pir::Int32Attribute"},
    {"float", "pir::FloatAttribute"},
    {"int64_t", "pir::Int64Attribute"},
    {"std::string", "pir::StrAttribute"},
    {"std::vector<int>", "pir::ArrayAttribute<pir::Int32Attribute>"},
    {"std::vector<float>", "pir::ArrayAttribute<pir::FloatAttribute>"},
    {"std::vector<int64_t>", "pir::ArrayAttribute<pir::Int64Attribute>"},
    {"std::vector<std::string>", "pir::ArrayAttribute<pir::StrAttribute>"}};
struct CombineOpInferSymbolicShapeInterfaceModel
    : public InferSymbolicShapeInterface::Concept {
  static inline bool InferSymbolicShape(
      pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
    std::vector<symbol::DimExpr> out_dims;

    // Currently for all operand : type.dims == 1u
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto type =
          op->operand(i).type().dyn_cast<paddle::dialect::DenseTensorType>();
      IR_ENFORCE(type, "Currently only support DenseTensorType.");
      IR_ENFORCE(type.dims().size() == 0u,
                 "Currently CombineOp only support 0-d DenseTensorType for "
                 "InferSymbolicShape. But the dims of the %d-th "
                 "DenseTensorType is %d.",
                 i,
                 type.dims().size());
    }

    auto operand_source_1st_data =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0)).data();
    if (operand_source_1st_data.has_value()) {
      for (auto operand_source : op->operands_source()) {
        auto source_data =
            shape_analysis->GetShapeOrDataForValue(operand_source)
                .data()
                .value();
        out_dims.push_back(source_data[0]);
      }
    }

    // TODO(zhangbopd): use op->result(0) to infer the shape
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(out_dims)};

    if (operand_source_1st_data.has_value()) {
      std::vector<symbol::DimExpr> tmp_shape(std::int64_t(out_dims.size()));
      symbol::ShapeOrDataDimExprs temp_shape_data{
          symbol::TensorShapeOrDataDimExprs(tmp_shape, out_dims)};
      shape_data = temp_shape_data;
    }

    op->set_attribute("symbolic_shape",
                      pir::shape::SymbolAttribute::get(
                          pir::IrContext::Instance(), shape_data));
    auto res = op->result(0);
    shape_analysis->SetShapeOrDataForValue(res, shape_data);
    return true;
  }

  CombineOpInferSymbolicShapeInterfaceModel()
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

    op->set_attribute("symbolic_shape",
                      pir::shape::SymbolAttribute::get(
                          pir::IrContext::Instance(), shape_data));

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
    op->set_attribute("symbolic_shape",
                      pir::shape::SymbolAttribute::get(
                          pir::IrContext::Instance(), shape_data));
    return true;
  }

  ShadowOutputOpInferSymbolicShapeInterfaceModel()
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
}

void PrintTypeImpl(pir::Type type, std::ostream& os) {
  os << type.dialect().name();
  os << '.';
  if (auto tensor_type = type.dyn_cast<DenseTensorType>()) {
    os << "tensor<";
    for (auto d : common::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  } else if (auto selected_rows_type = type.dyn_cast<SelectedRowsType>()) {
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
    pir::PrintInterleave(
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
  } else {
    printer.PrintGeneralOperation(op);
  }
}

void OperatorDialect::initialize() {
  RegisterTypes<paddle::dialect::DenseTensorType,
                paddle::dialect::SelectedRowsType,
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

  RegisterInterfaces<ParameterConvertInterface>();
}

void OperatorDialect::PrintType(pir::Type type, std::ostream& os) const {
  PrintTypeImpl(type, os);
}

void OperatorDialect::PrintAttribute(pir::Attribute attr,
                                     std::ostream& os) const {
  PrintAttributeImpl(attr, os);
}

pir::Type OperatorDialect::ParseType(pir::IrParser& parser) {  // NOLINT
  parser.ConsumeAToken("pd_op.tensor");
  parser.ConsumeAToken("<");
  std::vector<int> dim{};
  Token dim_token = parser.PeekToken();
  while (dim_token.token_type_ == DIGIT) {
    dim_token = parser.ConsumeToken();
    dim.push_back(atoi(dim_token.val_.c_str()));
    std::string peek_token_val = parser.PeekToken().val_;
    if (peek_token_val[0] != 'x') {
      break;
    }
    parser.ConsumeToken();
    parser.lexer->Unget(static_cast<int>(peek_token_val.size() - 1));
    if (parser.PeekToken().token_type_ != DIGIT) {
      break;
    }
  }
  phi::DDim ddim = common::make_ddim(dim);
  pir::Type dtype = parser.ParseType();
  std::vector<std::vector<size_t>> lod;
  std::vector<size_t> lodv;
  lodv.push_back(0);
  lod.push_back(lodv);
  parser.ConsumeAToken(">");
  return DenseTensorType::get(
      parser.ctx, dtype, ddim, phi::DataLayout::UNDEFINED, lod, 0);
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

void OperatorDialect::PrintOperation(pir::Operation* op,
                                     pir::IrPrinter& printer) const {
  PrintOperationImpl(op, printer);
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
    std::vector<paddle::dialect::OpInputInfo> inputs_info;
    std::vector<paddle::dialect::OpAttributeInfo> attributes_info;
    std::vector<paddle::dialect::OpOutputInfo> outputs_info;
    std::vector<std::string> param_names;
    // translate input info
    auto& op_input_names = OpMetaInfoHelper::GetInputs(op_meta);
    for (const auto& input_name : op_input_names) {
      param_names.push_back(input_name);
      bool is_optional = false;
      std::string input_type = "paddle::dialect::DenseTensorType";
      if (paddle::framework::detail::IsOptionalVar(input_name)) {
        is_optional = true;
      }
      if (paddle::framework::detail::IsDuplicableVar(input_name)) {
        input_type = "pir::VectorType<paddle::dialect::DenseTensorType>";
      }
      // Now, we only support dense tensor as input.
      inputs_info.push_back(paddle::dialect::OpInputInfo{
          input_name, input_type, is_optional, false, false, false});
    }

    // translate attr info
    auto& op_attrs = OpMetaInfoHelper::GetAttrs(op_meta);
    for (const auto& op_attr : op_attrs) {
      auto attr_name_and_type = paddle::ParseAttrStr(op_attr);
      auto attr_name = attr_name_and_type[0];
      auto attr_type_str = attr_name_and_type[1];
      param_names.push_back(attr_name);
      if (kCustomTypeMap.find(attr_type_str) == kCustomTypeMap.end()) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, "
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_type_str));
      }
      std::string attr_pir_type = kCustomTypeMap[attr_type_str];
      attributes_info.push_back(
          paddle::dialect::OpAttributeInfo{attr_name, attr_pir_type, ""});
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
    for (auto inplace_map : inplace_maps) {
      vec_inplace.push_back(inplace_map);
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

CustomOpDialect::CustomOpDialect(pir::IrContext* context)
    : pir::Dialect(name(), context, pir::TypeId::get<CustomOpDialect>()) {}

void CustomOpDialect::PrintType(pir::Type type, std::ostream& os) const {
  PrintTypeImpl(type, os);
}

void CustomOpDialect::PrintAttribute(pir::Attribute attr,
                                     std::ostream& os) const {
  PrintAttributeImpl(attr, os);
}

void CustomOpDialect::PrintOperation(pir::Operation* op,
                                     pir::IrPrinter& printer) const {
  PrintOperationImpl(op, printer);
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
