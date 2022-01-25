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

#include <fstream>
#include <iostream>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/common/string.h"
#include "paddle/infrt/dialect/basic_kernels.h"
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/init_infrt_dialects.h"
#include "paddle/infrt/dialect/pd_ops.h"
#include "paddle/infrt/dialect/tensor_shape.h"
#include "paddle/infrt/paddle/model_parser.h"

namespace cl = llvm::cl;
static cl::opt<std::string> paddleModelDir(cl::Positional,
                                           cl::desc("<paddle model dir>"),
                                           cl::init(""),
                                           cl::value_desc("model dir"));

// convert between std::vector and protobuf repeated.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(
      repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
  return ret;
}

int ConvertDataType(infrt::paddle::framework_proto::VarType::Type dtype,
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

class OpGenImpl {
 public:
  explicit OpGenImpl(mlir::MLIRContext *context)
      : context_(context), builder_(context) {
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
    InitHandlerMap();
  }
  mlir::ModuleOp ImportPaddleModel(
      const infrt::paddle::framework_proto::ProgramDesc &program) {
    main_block_ = program.blocks(0);

    llvm::SmallVector<mlir::Type, 4> operandTypes;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    // update inputs and outputs
    operandTypes.push_back(infrt::dt::TensorMapType::get(context_));
    for (auto &op_desc : main_block_.ops()) {
      if (op_desc.type() != "feed" && op_desc.type() != "fetch") {
        continue;
      }
      if (op_desc.type() == "fetch") {
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
      } else {
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
    }

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

    // update inputs
    operandTypes.push_back(infrt::dt::TensorMapType::get(context_));
    for (auto &op_desc : main_block_.ops()) {
      if (op_desc.type() == "feed") {
        for (int var_idx = 0; var_idx < op_desc.outputs_size(); ++var_idx) {
          // update input variables
          auto &in = op_desc.outputs()[var_idx];
          std::string input_var_name = in.arguments(0);
          ::mlir::Value input_ = mainFunc.getArgument(1);
          params_map_.insert(
              std::pair<std::string, mlir::Value>(input_var_name, input_));
        }
      }
    }

    // params
    ::mlir::Value map = mainFunc.getArgument(0);
    // update persistable tensors
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

    for (auto &op_desc : main_block_.ops()) {
      if (op_desc.type() == "feed" || op_desc.type() == "fetch") {
        continue;
      }
      (this->*(import_handler_map_[op_desc.type()]))(op_desc);
    }

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
    module_->dump();
    return module_;
  }

  template <typename T>
  void buildOperation(const infrt::paddle::framework_proto::OpDesc &op_) {
    auto op_name = T::getOperationName();
    mlir::Location loc = mlir::UnknownLoc::get(context_);

    llvm::SmallVector<mlir::Value, 4> operands;
    // op inputs
    for (int var_idx = 0; var_idx < op_.inputs_size(); ++var_idx) {
      auto &var = op_.inputs(var_idx);
      if (!var.arguments().empty()) {
        operands.push_back((params_map_[var.arguments()[0]]));
      }
    }

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    // update op outputs info
    for (int var_idx = 0; var_idx < op_.outputs_size(); ++var_idx) {
      auto &var_name = op_.outputs(var_idx).arguments()[0];
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

    // GetInputVarName
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    // update attrs
    for (int attrs_num = 0; attrs_num < op_.attrs_size(); attrs_num++) {
      auto attr_name_ = op_.attrs(attrs_num).name();
      auto type = op_.attrs(attrs_num).type();
      if (std::count(skipped_attrs_.begin(), skipped_attrs_.end(), attr_name_))
        continue;
      switch (type) {
        case infrt::paddle::framework_proto::AttrType::FLOAT: {
          float data = op_.attrs(attrs_num).f();
          auto value_ = builder_.getF32FloatAttr(data);
          auto name_ = builder_.getStringAttr(attr_name_);
          auto attr_ = mlir::NamedAttribute(name_, value_);
          attrs.push_back(attr_);
          break;
        }
        // bool
        case infrt::paddle::framework_proto::AttrType::BOOLEAN: {
          bool data = op_.attrs(attrs_num).b();
          auto value_ = builder_.getBoolAttr(data);
          auto name_ = builder_.getStringAttr(attr_name_);
          auto attr_ = mlir::NamedAttribute(name_, value_);
          attrs.push_back(attr_);
          break;
        }
        // int
        case infrt::paddle::framework_proto::AttrType::INT: {
          int data = op_.attrs(attrs_num).i();
          auto value_ = builder_.getI32IntegerAttr(data);
          auto name_ = builder_.getStringAttr(attr_name_);
          auto attr_ = mlir::NamedAttribute(name_, value_);
          attrs.push_back(attr_);
          break;
        }
        // long
        case infrt::paddle::framework_proto::AttrType::LONG: {
          int64_t data = op_.attrs(attrs_num).l();
          auto value_ = builder_.getI64IntegerAttr(data);
          auto name_ = builder_.getStringAttr(attr_name_);
          auto attr_ = mlir::NamedAttribute(name_, value_);
          attrs.push_back(attr_);
          break;
        }
        case infrt::paddle::framework_proto::AttrType::STRING: {
          std::string data = op_.attrs(attrs_num).s();
          auto value_ = builder_.getStringAttr(data);
          auto name_ = builder_.getStringAttr(attr_name_);
          auto attr_ = mlir::NamedAttribute(name_, value_);
          attrs.push_back(attr_);
          break;
        }
        case infrt::paddle::framework_proto::AttrType::BLOCK: {
          //            auto data = op_.attrs(attrs_num).block_idx();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        case infrt::paddle::framework_proto::AttrType::INTS: {
          //            auto data = op_.attrs(attrs_num).ints();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        case infrt::paddle::framework_proto::AttrType::FLOATS: {
          //            auto data = op_.attrs(attrs_num).floats();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        case infrt::paddle::framework_proto::AttrType::STRINGS: {
          //            auto data = op_.attrs(attrs_num).strings();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        case infrt::paddle::framework_proto::AttrType::BOOLEANS: {
          //            auto data = op_.attrs(attrs_num).bools();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }

        case infrt::paddle::framework_proto::AttrType::BLOCKS: {
          //            auto data = op_.attrs(attrs_num).blocks_idx();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        case infrt::paddle::framework_proto::AttrType::LONGS: {
          //            auto data = op_.attrs(attrs_num).longs();
          //            cout << attr_name_ <<" data:" << data << std::endl;
          break;
        }
        default:
          std::cout << "error attribute\n";
      }
    }

    // op inputs
    for (int var_idx = 0; var_idx < op_.inputs_size(); ++var_idx) {
      auto &var = op_.inputs(var_idx);
      if (!var.arguments().empty()) {
        operands.push_back((params_map_[var.arguments()[0]]));
      }
    }

    mlir::OperationState result(loc, op_name, operands, resultTypes, attrs);
    auto *mlir_op_ = builder_.createOperation(result);
    // op outputs
    for (int var_idx = 0; var_idx < op_.outputs_size(); ++var_idx) {
      auto &var_name = op_.outputs(var_idx).arguments()[0];
      // output name
      auto var_ = mlir_op_->getResult(var_idx);
      params_map_.insert(std::pair<std::string, mlir::Value>(var_name, var_));
    }
  }

 private:
  void InitHandlerMap() {
#include "tools/infrt/OpBuildTable.inc"
  }
  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  std::map<std::string, mlir::Value> params_map_;
  infrt::paddle::framework_proto::BlockDesc main_block_;
  using ImportHandlerType =
      void (OpGenImpl::*)(const infrt::paddle::framework_proto::OpDesc &op_);
  std::map<std::string, ImportHandlerType> import_handler_map_;
  const std::vector<std::string> skipped_attrs_{"trainable_statistics",
                                                "use_global_stats",
                                                "is_test",
                                                "use_mkldnn",
                                                "use_cudnn",
                                                "op_device",
                                                "op_namescope",
                                                "op_role",
                                                "mkldnn_data_type",
                                                "with_quant_attr",
                                                "use_quantizer"};
};

int main(int argc, char **argv) {
  // parse paddle model path
  cl::ParseCommandLineOptions(argc, argv, "paddle-mlir");
  if (paddleModelDir.empty()) {
    std::cout << "ERROR: paddle model path can't be empty." << std::endl;
    return 1;
  }

  // basic configs
  infrt::paddle::Scope scope;
  infrt::common::Target target;
  target.arch = infrt::common::Target::Arch::X86;
  target.bits = infrt::common::Target::Bit::k32;
  target.os = infrt::common::Target::OS::Linux;
  std::string model_path = paddleModelDir + "/__model__";
  std::string params_path = paddleModelDir + "/params";

  // Load program description
  auto program_proto = *infrt::paddle::LoadProgram(model_path);

  // load params
  auto main_block = program_proto.blocks(0);
  std::vector<std::string> paramlist;
  for (auto &var : main_block.vars()) {
    // Get vars
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable())
      continue;
    paramlist.push_back(var.name());
  }
  std::stable_sort(paramlist.begin(), paramlist.end());

  // Load vars
  auto load_var_func = [&](std::istream &is) {
    for (size_t i = 0; i < paramlist.size(); ++i) {
      auto *var = scope.Var<infrt::paddle::Tensor>(paramlist[i]);
      // Error checking
      CHECK(static_cast<bool>(is))
          << "There is a problem with loading model parameters";
      LoadLoDTensor(is, var, target);
    }
    is.peek();
    CHECK(is.eof()) << "You are not allowed to load partial data via"
                    << " LoadCombinedParamsPb, use LoadParam instead.";
  };

  std::ifstream fin(params_path, std::ios::binary);
  CHECK(fin.is_open());
  load_var_func(fin);

  // load infrt dialects
  mlir::MLIRContext *context = infrt::Global::getMLIRContext();
  context->allowUnregisteredDialects();
  context->getOrLoadDialect<mlir::StandardOpsDialect>();
  context->getOrLoadDialect<infrt::dialect::INFRTDialect>();
  context->getOrLoadDialect<infrt::ts::TensorShapeDialect>();
  context->getOrLoadDialect<infrt::dt::DTDialect>();
  context->getOrLoadDialect<mlir::pd::PaddleDialect>();

  OpGenImpl myGen(context);
  myGen.ImportPaddleModel(program_proto);
}
