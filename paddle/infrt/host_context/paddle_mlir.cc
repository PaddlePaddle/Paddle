// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/host_context/paddle_mlir.h"
#include "paddle/infrt/dialect/pd_ops_info.h"

MLIRModelGenImpl::MLIRModelGenImpl()
    : context_(infrt::Global::getMLIRContext()), builder_(context_) {
  context_->allowUnregisteredDialects();
  context_->getOrLoadDialect<mlir::StandardOpsDialect>();
  context_->getOrLoadDialect<infrt::dialect::INFRTDialect>();
  context_->getOrLoadDialect<infrt::ts::TensorShapeDialect>();
  context_->getOrLoadDialect<infrt::dt::DTDialect>();
  context_->getOrLoadDialect<mlir::pd::PaddleDialect>();
  module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(context_));
}

infrt::paddle::framework_proto::ProgramDesc MLIRModelGenImpl::ParsePaddleModel(
    const std::string &model_file) {
  infrt::paddle::framework_proto::ProgramDesc program_proto =
      *infrt::paddle::LoadProgram(model_file);
  return program_proto;
}

mlir::ModuleOp MLIRModelGenImpl::ImportPaddleModel(
    const std::string &model_dir) {
  infrt::paddle::framework_proto::ProgramDesc program_proto =
      ParsePaddleModel(model_dir + "/__model__");
  return ImportPaddleModel(program_proto);
}

mlir::ModuleOp MLIRModelGenImpl::ImportPaddleModel(
    const std::string &model_file, const std::string &param_file) {
  infrt::paddle::framework_proto::ProgramDesc program_proto =
      ParsePaddleModel(model_file);
  return ImportPaddleModel(program_proto);
}

mlir::ModuleOp MLIRModelGenImpl::ImportPaddleModel(
    const infrt::paddle::framework_proto::ProgramDesc &program) {
  main_block_ = program.blocks(0);
  llvm::SmallVector<mlir::Type, 4> operandTypes = GetModelInputsType(program);
  llvm::SmallVector<mlir::Type, 4> resultTypes = GetModelOutputsType(program);
  mlir::FuncOp mainFunc = UpdateModelModule(operandTypes, resultTypes);
  UpdateModelParams(program, &mainFunc);
  UpdateModelOps(program);
  UpdateModelOutputs(program);
  return module_;
}

mlir::FuncOp MLIRModelGenImpl::UpdateModelModule(
    llvm::SmallVector<mlir::Type, 4> operandTypes,
    llvm::SmallVector<mlir::Type, 4> resultTypes) {
  // create main op
  const std::string &name = "main_graph";
  auto mainFunc = mlir::FuncOp::create(
      mlir::UnknownLoc::get(context_),
      name,
      /*type=*/builder_.getFunctionType({operandTypes}, {resultTypes}),
      /*attrs=*/{});
  module_.push_back(mainFunc);
  mainFunc.addEntryBlock();
  builder_.setInsertionPointToStart(&mainFunc.body().back());
  return mainFunc;
}

llvm::SmallVector<mlir::Type, 4> MLIRModelGenImpl::GetModelInputsType(
    const infrt::paddle::framework_proto::ProgramDesc &program) {
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  operandTypes.push_back(infrt::dt::TensorMapType::get(context_));
  for (auto &op_desc : main_block_.ops()) {
    if (op_desc.type() != "feed") continue;
    for (int var_idx = 0; var_idx < op_desc.outputs_size(); ++var_idx) {
      // update input variables
      auto &in = op_desc.outputs()[var_idx];
      std::string input_var_name = in.arguments(0);
      for (int i = 0; i < main_block_.vars_size(); i++) {
        auto var_desc = main_block_.vars(i);
        if (var_desc.name() == input_var_name) {
          std::vector<int64_t> dims = RepeatedToVector<int64_t>(
              var_desc.type().lod_tensor().tensor().dims());
          mlir::Type precision_;
          ConvertDataType(var_desc.type().lod_tensor().tensor().data_type(),
                          builder_,
                          &precision_);
          mlir::Type type_ = mlir::RankedTensorType::get(dims, precision_);
          operandTypes.push_back(type_);
        }
      }
    }
  }
  return operandTypes;
}

llvm::SmallVector<mlir::Type, 4> MLIRModelGenImpl::GetModelOutputsType(
    const infrt::paddle::framework_proto::ProgramDesc &program) {
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (auto &op_desc : main_block_.ops()) {
    if (op_desc.type() != "fetch") continue;
    for (int var_idx = 0; var_idx < op_desc.inputs_size(); ++var_idx) {
      auto &in = op_desc.inputs()[var_idx];
      std::string input_var_name = in.arguments(0);
      for (int i = 0; i < main_block_.vars_size(); i++) {
        auto var_desc = main_block_.vars(i);
        if (var_desc.name() == input_var_name) {
          std::vector<int64_t> dims = RepeatedToVector<int64_t>(
              var_desc.type().lod_tensor().tensor().dims());
          mlir::Type precision_;
          ConvertDataType(var_desc.type().lod_tensor().tensor().data_type(),
                          builder_,
                          &precision_);
          mlir::Type type_ = mlir::RankedTensorType::get(dims, precision_);
          resultTypes.push_back(type_);
        }
      }
    }
  }
  return resultTypes;
}

void MLIRModelGenImpl::UpdateModelOps(
    const infrt::paddle::framework_proto::ProgramDesc &program) {
  for (auto &op_desc : main_block_.ops()) {
    if (op_desc.type() == "feed" || op_desc.type() == "fetch") {
      continue;
    }
    buildOperation(op_desc);
  }
}

void MLIRModelGenImpl::UpdateModelParams(
    const infrt::paddle::framework_proto::ProgramDesc &program,
    mlir::FuncOp *mainFunc) {
  // update input vars
  for (auto &op_desc : main_block_.ops()) {
    if (op_desc.type() == "feed") {
      for (int var_idx = 0; var_idx < op_desc.outputs_size(); ++var_idx) {
        // update input variables
        auto &in = op_desc.outputs()[var_idx];
        std::string input_var_name = in.arguments(0);
        ::mlir::Value input_ = mainFunc->getArgument(1);
        params_map_.insert(
            std::pair<std::string, mlir::Value>(input_var_name, input_));
      }
    }
  }

  // update persistable tensors
  ::mlir::Value map = mainFunc->getArgument(0);
  for (int i = 0; i < main_block_.vars_size(); i++) {
    auto var_desc = main_block_.vars(i);
    if (params_map_.find(var_desc.name()) != params_map_.end()) continue;
    if (var_desc.name() != "feed" && var_desc.name() != "fetch" &&
        var_desc.persistable()) {
      auto name = builder_.getStringAttr(var_desc.name());
      std::vector<int64_t> dims = RepeatedToVector<int64_t>(
          var_desc.type().lod_tensor().tensor().dims());
      mlir::Type precision_;
      ConvertDataType(var_desc.type().lod_tensor().tensor().data_type(),
                      builder_,
                      &precision_);
      mlir::Type type_ = mlir::RankedTensorType::get(dims, precision_);
      auto op = builder_.create<infrt::dt::GetParamOp>(
          mlir::UnknownLoc::get(context_), type_, map, name);
      params_map_.insert(std::pair<std::string, mlir::Value>(
          var_desc.name(), op.getOperation()->getResult(0)));
    }
  }
}

void MLIRModelGenImpl::UpdateModelOutputs(
    const infrt::paddle::framework_proto::ProgramDesc &program) {
  // update outputs
  for (auto &op_desc : main_block_.ops()) {
    if (op_desc.type() == "fetch") {
      for (int var_idx = 0; var_idx < op_desc.inputs_size(); ++var_idx) {
        auto &in = op_desc.inputs()[var_idx];
        // varibale name
        std::string input_var_name = in.arguments(0);
        // update model outpus
        mlir::Location loc = mlir::UnknownLoc::get(context_);
        llvm::SmallVector<mlir::Value, 4> operands;

        operands.push_back((params_map_[input_var_name]));

        llvm::SmallVector<mlir::Type, 4> resultTypes;
        llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
        mlir::OperationState state(loc,
                                   mlir::ReturnOp::getOperationName(),
                                   operands,
                                   resultTypes,
                                   attrs);
        builder_.createOperation(state);
      }
    }
  }
}

void MLIRModelGenImpl::buildOperation(
    const infrt::paddle::framework_proto::OpDesc &op_) {
  const std::string &op_name = "pd." + op_.type();
  mlir::Location loc = mlir::UnknownLoc::get(context_);

  llvm::SmallVector<mlir::Value, 4> operands = GetOpInputValue(op_);
  llvm::SmallVector<mlir::Type, 4> resultTypes = GetOpOutputType(op_);
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs = GetOpAttributes(op_);
  mlir::OperationState result(loc, op_name, operands, resultTypes, attrs);
  mlir::Operation *mlir_op_ = builder_.createOperation(result);
  RegisterOpOutputVars(op_, mlir_op_);
}

llvm::SmallVector<mlir::Value, 4> MLIRModelGenImpl::GetOpInputValue(
    const infrt::paddle::framework_proto::OpDesc &op_) {
  llvm::SmallVector<mlir::Value, 4> operands;

  std::vector<std::string> inputs_info = {};
  if (pd_dialect_inputs_info_map_.count(op_.type()))
    inputs_info = pd_dialect_inputs_info_map_.at(op_.type());

  for (int var_idx = 0; var_idx < op_.inputs_size(); ++var_idx) {
    auto &var = op_.inputs(var_idx);
    if (!var.arguments().empty()) {
      if (!std::count(inputs_info.begin(), inputs_info.end(), var.parameter()))
        continue;
      operands.push_back((params_map_[var.arguments()[0]]));
    }
  }
  return operands;
}

llvm::SmallVector<mlir::Type, 4> MLIRModelGenImpl::GetOpOutputType(
    const infrt::paddle::framework_proto::OpDesc &op_) {
  llvm::SmallVector<mlir::Type, 4> resultTypes;

  std::vector<std::string> pd_dialect_outputs_info = {};
  if (pd_dialect_outputs_info_map_.count(op_.type()))
    pd_dialect_outputs_info = pd_dialect_outputs_info_map_.at(op_.type());

  // update op outputs info
  for (int var_idx = 0; var_idx < op_.outputs_size(); ++var_idx) {
    auto &var_name = op_.outputs(var_idx).arguments()[0];

    if (!std::count(pd_dialect_outputs_info.begin(),
                    pd_dialect_outputs_info.end(),
                    op_.outputs(var_idx).parameter()))
      continue;

    // update persistable tensors
    for (int i = 0; i < main_block_.vars_size(); i++) {
      auto var_desc = main_block_.vars(i);
      if (var_desc.name() == var_name) {
        std::vector<int64_t> dims = RepeatedToVector<int64_t>(
            var_desc.type().lod_tensor().tensor().dims());
        mlir::Type precision_;
        ConvertDataType(var_desc.type().lod_tensor().tensor().data_type(),
                        builder_,
                        &precision_);
        mlir::Type type_ = mlir::RankedTensorType::get(dims, precision_);
        resultTypes.push_back(type_);
      }
    }
  }
  return resultTypes;
}

llvm::SmallVector<mlir::NamedAttribute, 4> MLIRModelGenImpl::GetOpAttributes(
    const infrt::paddle::framework_proto::OpDesc &op_) {
  // GetInputVarName
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs;

#define ATTR_IMPL_CASE(PROTO_TYPE, PROTO_TYPE_METHOD, MLIR_TYPE_METHOD) \
  case infrt::paddle::framework_proto::AttrType::PROTO_TYPE: {          \
    auto data = op_.attrs(attrs_num).PROTO_TYPE_METHOD();               \
    auto value_ = builder_.MLIR_TYPE_METHOD(data);                      \
    auto name_ = builder_.getStringAttr(attr_name_);                    \
    auto attr_ = mlir::NamedAttribute(name_, value_);                   \
    attrs.push_back(attr_);                                             \
    break;                                                              \
  }

#define REPEATED_ATTR_IMPLE_CASE(                                       \
    PROTO_TYPE, PROTO_TYPE_METHOD, MLIR_TYPE, MLIR_TYPE_METHOD)         \
  case infrt::paddle::framework_proto::AttrType::PROTO_TYPE: {          \
    std::vector<MLIR_TYPE> data;                                        \
    for (const auto &var : op_.attrs(attrs_num).PROTO_TYPE_METHOD()) {  \
      data.push_back(MLIR_TYPE(var));                                   \
    }                                                                   \
    auto value_ =                                                       \
        builder_.MLIR_TYPE_METHOD(llvm::makeArrayRef<MLIR_TYPE>(data)); \
    auto name_ = builder_.getStringAttr(attr_name_);                    \
    auto attr_ = mlir::NamedAttribute(name_, value_);                   \
    attrs.push_back(attr_);                                             \
    break;                                                              \
  }

#define UNIMPLEMENTED_ATTR_IMPL_CASE(PROTO_TYPE)                        \
  case infrt::paddle::framework_proto::AttrType::PROTO_TYPE: {          \
    std::cout << "Unimplemented attr type: framework_proto::AttrType::" \
              << #PROTO_TYPE << std::endl;                              \
    break;                                                              \
  }

  // get registered attributes
  const std::string &op_name = "pd." + op_.type();
  mlir::RegisteredOperationName registered_op_name_ =
      mlir::RegisteredOperationName::lookup(op_name, context_).getValue();
  llvm::ArrayRef<mlir::StringAttr> attr_names_ =
      registered_op_name_.getAttributeNames();
  std::vector<mlir::StringAttr> attr_names_vec_ = attr_names_.vec();

  // update attrs
  for (int attrs_num = 0; attrs_num < op_.attrs_size(); attrs_num++) {
    auto attr_name_ = op_.attrs(attrs_num).name();
    auto type = op_.attrs(attrs_num).type();
    if (!std::count(attr_names_vec_.begin(), attr_names_vec_.end(), attr_name_))
      continue;
    switch (type) {
      ATTR_IMPL_CASE(FLOAT, f, getF32FloatAttr);
      ATTR_IMPL_CASE(BOOLEAN, b, getBoolAttr);
      ATTR_IMPL_CASE(INT, i, getI32IntegerAttr);
      ATTR_IMPL_CASE(LONG, l, getI64IntegerAttr);
      ATTR_IMPL_CASE(STRING, s, getStringAttr);

      REPEATED_ATTR_IMPLE_CASE(
          STRINGS, strings, llvm::StringRef, getStrArrayAttr);
      REPEATED_ATTR_IMPLE_CASE(FLOATS, floats, float, getF32ArrayAttr);
      REPEATED_ATTR_IMPLE_CASE(INTS, ints, int32_t, getI32ArrayAttr);
      REPEATED_ATTR_IMPLE_CASE(LONGS, longs, int64_t, getI64ArrayAttr);

      // Unimplemented attr type, will be supported later @DannyIsFunny
      // bools attribute is not supported due to bug of llvm.
      // REPEATED_ATTR_IMPLE_CASE(BOOLEANS, bools, bool, getBoolArrayAttr);
      UNIMPLEMENTED_ATTR_IMPL_CASE(BOOLEANS);
      UNIMPLEMENTED_ATTR_IMPL_CASE(BLOCK);
      UNIMPLEMENTED_ATTR_IMPL_CASE(BLOCKS);
      default:
        std::cout << "error attribute" << attr_name_ << std::endl;
    }
  }
  return attrs;
}

void MLIRModelGenImpl::RegisterOpOutputVars(
    const infrt::paddle::framework_proto::OpDesc &op_,
    mlir::Operation *mlir_op_) {
  // op outputs
  for (int var_idx = 0; var_idx < op_.outputs_size(); ++var_idx) {
    auto &var_name = op_.outputs(var_idx).arguments()[0];
    // output name
    auto var_ = mlir_op_->getResult(var_idx);
    params_map_.insert(std::pair<std::string, mlir::Value>(var_name, var_));
  }
}

bool ConvertDataType(infrt::paddle::framework_proto::VarType::Type dtype,
                     mlir::Builder builder,
                     mlir::Type *type) {
  switch (dtype) {
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_FP16:
      *type = builder.getF16Type();
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_FP32:
      *type = builder.getF32Type();
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_FP64:
      *type = builder.getF64Type();
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_BOOL:
      *type = builder.getIntegerType(1);
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_INT8:
      *type = builder.getIntegerType(8);
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_INT16:
      *type = builder.getIntegerType(16);
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_INT32:
      *type = builder.getIntegerType(32);
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_INT64:
      *type = builder.getIntegerType(64);
      return true;
    case infrt::paddle::framework_proto::VarType::Type::VarType_Type_UINT8:
      *type = builder.getIntegerType(8, /*isSigned=*/false);
      return true;
    default:
      return false;
  }
}
