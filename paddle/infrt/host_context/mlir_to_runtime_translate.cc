// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "boost/optional.hpp"
#include "paddle/infrt/common/string.h"
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/dialect/tensor_shape.h"
#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_frame.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"
#include "paddle/infrt/host_context/op_executable.h"
#include "paddle/infrt/host_context/value.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt::host_context {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

struct MlirToRuntimeTranslator::Impl {
  mlir::ModuleOp module;
  // The runtime for a function call.
  CoreRuntimeBuilder* runtime{};
  // The current working op, the translator process the ops one by one, each
  // time it updates `cur_op` here to current op
  // working on.
  OpExecutableBuilder* cur_op{};

  // record the current function name.
  std::string cur_func_name;

  // Name to function definitions.
  std::unordered_map<std::string, mlir::FuncOp> func_defs;

  // Map from an operation to its results.
  std::unordered_map<const mlir::Operation*, std::vector<ValueRef>> op_results;
  llvm::DenseMap<mlir::Value, ValueRef> value_map;
};

bool MlirToRuntimeTranslator::EmitConstantOp(mlir::Operation* op) {
  if (!infrt::Startswith(op->getName().getStringRef().str(), "infrt.constant"))
    return false;
  VLOG(3) << "Emitting constant op [" << op->getName().getStringRef().str()
          << "]";

  auto attr = op->getAttr("value");
  if (attr.isa<mlir::FloatAttr>()) {
    if (attr.getType().isF32()) {
      impl_->op_results[op] = {ValueRef(
          static_cast<float>(attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else if (attr.getType().isF64()) {
      impl_->op_results[op] = {ValueRef(static_cast<double>(
          attr.cast<mlir::FloatAttr>().getValueAsDouble()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  if (attr.isa<mlir::IntegerAttr>()) {
    if (attr.getType().isInteger(32)) {
      impl_->op_results[op] = {ValueRef(
          static_cast<int32_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(64)) {
      impl_->op_results[op] = {ValueRef(
          static_cast<int64_t>(attr.cast<mlir::IntegerAttr>().getSInt()))};
    } else if (attr.getType().isInteger(1)) {
      impl_->op_results[op] = {
          ValueRef(static_cast<bool>(attr.cast<mlir::IntegerAttr>().getInt()))};
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
    return true;
  }

  LOG(FATAL) << "Not supported constant attribute type";
  return true;
}

template <>
boost::optional<int32_t> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::IntegerAttr>()) return boost::none;
  if (attr->isa<mlir::IntegerAttr>()) {
    auto val = attr->cast<mlir::IntegerAttr>();
    if (val.getType().isInteger(32)) {
      return val.getInt();
    }
  }
  return boost::none;
}
template <>
boost::optional<int64_t> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::IntegerAttr>()) return boost::none;
  if (attr->isa<mlir::IntegerAttr>()) {
    auto val = attr->cast<mlir::IntegerAttr>();
    if (val.getType().isInteger(64)) {
      return val.getInt();
    }
  }
  return boost::none;
}

// TODO(Superjomn) Make double and float parsing share some thing.
template <>
boost::optional<float> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::FloatAttr>()) return boost::none;
  if (attr->isa<mlir::FloatAttr>()) {
    auto val = attr->cast<mlir::FloatAttr>();
    if (val.getType().isF32()) return val.getValueAsDouble();
  }
  return boost::none;
}

template <>
boost::optional<double> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::FloatAttr>()) return boost::none;
  if (attr->isa<mlir::FloatAttr>()) {
    auto val = attr->cast<mlir::FloatAttr>();
    if (val.getType().isF64()) return val.getValueAsDouble();
  }
  return boost::none;
}

template <>
boost::optional<std::string> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::StringAttr>()) return boost::none;
  return attr->cast<mlir::StringAttr>().getValue().str();
}

#define PROCESS_ARRAY_INT(type__, bits__)                                      \
  template <>                                                                  \
  boost::optional<std::vector<type__>> MlirToRuntimeTranslator::EmitAttribute( \
      const mlir::Attribute* attr) {                                           \
    if (!attr->isa<mlir::ArrayAttr>()) return boost::none;                     \
    auto array = attr->cast<mlir::ArrayAttr>();                                \
    CHECK(!array.empty());                                                     \
                                                                               \
    if (!array[0].getType().isInteger(bits__)) {                               \
      return boost::none;                                                      \
    }                                                                          \
                                                                               \
    std::vector<type__> res;                                                   \
    for (auto& v : array) {                                                    \
      res.push_back(v.cast<mlir::IntegerAttr>().getInt());                     \
    }                                                                          \
    return res;                                                                \
  }

PROCESS_ARRAY_INT(int16_t, 16);
PROCESS_ARRAY_INT(int32_t, 32);
PROCESS_ARRAY_INT(int64_t, 64);

template <>
boost::optional<std::vector<float>> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::ArrayAttr>()) return boost::none;
  auto array = attr->cast<mlir::ArrayAttr>();
  CHECK(!array.empty());

  if (!array[0].getType().isF32()) return boost::none;

  std::vector<float> res;
  for (auto& v : array) {
    res.push_back(v.cast<mlir::FloatAttr>().getValueAsDouble());
  }
  return res;
}

template <>
boost::optional<std::vector<double>> MlirToRuntimeTranslator::EmitAttribute(
    const mlir::Attribute* attr) {
  if (!attr->isa<mlir::ArrayAttr>()) return boost::none;
  auto array = attr->cast<mlir::ArrayAttr>();
  CHECK(!array.empty());

  if (!array[0].getType().isF64()) return boost::none;

  std::vector<double> res;
  for (auto& v : array) {
    res.push_back(v.cast<mlir::FloatAttr>().getValueAsDouble());
  }
  return res;
}

static bool IsReturn(mlir::Operation* op) {
  return op->getName().getStringRef() == "infrt.return";
}

bool MlirToRuntimeTranslator::EmitGeneralOp(mlir::Operation* op) {
  CHECK(impl_->runtime);
  impl_->cur_op =
      impl_->runtime->NewOpExecutable(op->getName().getStringRef().str());

  VLOG(3) << "processing general op : " << op->getName().getStringRef().str();

  // process operands
  for (int i = 0, e = op->getNumOperands(); i < e; i++) {
    // function argument as value
    auto operand = op->getOperand(i);
    if (operand.getKind() == mlir::Value::Kind::BlockArgument) {
      mlir::BlockArgument arg = operand.dyn_cast<mlir::BlockArgument>();
      Value* arg_value = GetValue(arg);
      impl_->cur_op->AppendArgument(arg_value);
      VLOG(3) << "* op mlir operand: " << DumpToString(arg) << " "
              << GetValue(arg);
      continue;
    }

    // normal value
    Value* arg_value = GetValue(operand);
    if (!arg_value) {
      auto upstream_op = operand.getDefiningOp();
      arg_value = GetOpResult(upstream_op);
    }
    CHECK(arg_value) << "No-exist argument value found: "
                     << DumpToString(operand);
    impl_->cur_op->AppendArgument(arg_value);

    VLOG(3) << "* op mlir operand: " << DumpToString(operand) << " "
            << GetValue(operand) << " vs " << arg_value;
  }

  // process results
  llvm::SmallVector<Value*, 4> res_values;
  for (int i = 0, e = op->getNumResults(); i < e; i++) {
    auto res = op->getResult(i);
    res_values.push_back(AddValue(res));

    VLOG(3) << "* op mlir res: " << DumpToString(res) << " " << GetValue(res);
  }
  impl_->cur_op->SetResults(res_values);

#ifdef INFRT_DEBUG
  {
    VLOG(3) << "check result";
    for (int i = 0; i < impl_->cur_op->frame().GetNumResults(); i++) {
      VLOG(3) << "+ res value: " << impl_->cur_op->frame().GetResults()[i];
    }
  }
#endif

  // process attributes
  auto attrs = op->getAttrs();

  for (size_t i = 0; i < attrs.size(); i++) {
    auto& attr = attrs[i];
    if (auto v = EmitAttribute<int32_t>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(*v));
    } else if (auto v = EmitAttribute<int64_t>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(*v));
    } else if (auto v = EmitAttribute<float>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(*v));
    } else if (auto v = EmitAttribute<double>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(*v));
    } else if (auto v = EmitAttribute<std::string>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else if (auto v = EmitAttribute<std::vector<int16_t>>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else if (auto v = EmitAttribute<std::vector<int32_t>>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else if (auto v = EmitAttribute<std::vector<int64_t>>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else if (auto v = EmitAttribute<std::vector<float>>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else if (auto v = EmitAttribute<std::vector<double>>(&attr.second)) {
      impl_->cur_op->AppendAttribute(new Value(std::move(*v)));
    } else {
      LOG(FATAL) << "Not supported attribute type";
    }
  }

  // process regions, we treat regions as attribute.
  auto num_regions = op->getNumRegions();
  if (num_regions > 0) {
    CHECK_EQ(num_regions, 1UL)
        << "op with more than one region is not supported yet.";
    auto& region = op->getRegions().front();
    auto num_blocks = region.getBlocks().size();
    CHECK_EQ(num_blocks, 1UL)
        << "region with more than one block is not supported yet.";

    // process arguments
    llvm::SmallVector<mlir::Type, 4> inputs;
    auto& block = region.getBlocks().front();
    for (auto arg : block.getArguments()) inputs.push_back(arg.getType());

    // process results
    // NOTE: if an op contains a region, we simply ignore the region's return
    // values,
    //       or its return values will conflict with op's return values.
    llvm::SmallVector<mlir::Type, 0> results;

    auto func_type =
        mlir::FunctionType::get(inputs, results, region.getContext());
    auto* function = impl_->cur_op->CreateFunctionExecutable(
        &region, func_type, &impl_->func_defs);
    impl_->cur_op->AppendAttribute(new Value(function));
  }

  return true;
}

bool MlirToRuntimeTranslator::EmitReturnOp(
    mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>* results) {
  CHECK(results);
  if (op->getName().getStringRef() == "infrt.return") {
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      results->push_back(op->getOperand(i));
    }

    return true;
  }
  return false;
}

bool MlirToRuntimeTranslator::EmitFunctions() {
  for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
    EmitFunction(func_op);
  }
  return true;
}

void MlirToRuntimeTranslator::EmitFunction(mlir::FuncOp op) {
  impl_->func_defs[op.getName().str()] = op;
}

Value* MlirToRuntimeTranslator::GetOpResult(mlir::Operation* op) {
  auto it = impl_->op_results.find(op);
  return it == impl_->op_results.end() ? nullptr : it->second.front().get();
}

Value* MlirToRuntimeTranslator::GetValue(mlir::Value value) {
  auto it = impl_->value_map.find(value);
  return it == impl_->value_map.end() ? nullptr : it->second.get();
}

Value* MlirToRuntimeTranslator::AddValue(mlir::Value value) {
  auto res = impl_->value_map.try_emplace(value, ValueRef(new Value));
  CHECK(res.second) << "Duplicate add mlir value [" << DumpToString(value)
                    << "]";
  return res.first->second.get();
}

MlirToRuntimeTranslator::~MlirToRuntimeTranslator() {}

void MlirToRuntimeTranslator::UpdateCurFuncName(const std::string& name) {
  impl_->cur_func_name = std::string(name);
}

MlirToRuntimeTranslator::MlirToRuntimeTranslator(mlir::ModuleOp module,
                                                 CoreRuntimeBuilder* runtime)
    : impl_(new Impl) {
  CHECK(runtime);
  impl_->module = module;
  impl_->runtime = runtime;
}

bool MlirToRuntimeTranslator::EmitBuildShapeOp(mlir::Operation* op) {
  if (op->getName().getStringRef() != "ts.build_shape") return false;

  auto value = op->getAttr("value");

  CHECK(value.isa<mlir::ArrayAttr>());
  auto values = value.cast<mlir::ArrayAttr>().getValue();
  std::vector<int64_t> dims;
  for (auto& attr_v : values) {
    dims.push_back(attr_v.cast<mlir::IntegerAttr>().getInt());
  }
  impl_->op_results[op] = {
      ValueRef(new Value(tensor::TensorShape(llvm::ArrayRef<int64_t>(dims))))};

  return true;
}

bool MlirToRuntimeTranslator::EmitCallOp(mlir::Operation* op,
                                         function_defs_t* function_table) {
  CHECK(op);
  CHECK(function_table);
  if (op->getName().getStringRef() != "infrt.call") return false;

  impl_->cur_op =
      impl_->runtime->NewOpExecutable(op->getName().getStringRef().str());

  auto callee = op->getAttr("callee");
  auto callee_name = callee.dyn_cast<mlir::FlatSymbolRefAttr>();

  // process arguments
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto operand = op->getOperand(i);
    auto* arg_value = GetValue(operand);

    if (!arg_value) {
      auto upstream_op = operand.getDefiningOp();
      arg_value = GetOpResult(upstream_op);
    }
    CHECK(arg_value) << "No-exist argument value found: "
                     << DumpToString(operand);
    impl_->cur_op->AppendArgument(arg_value);
  }

  // process results
  llvm::SmallVector<Value*, 4> res_values;
  for (int i = 0, e = op->getNumResults(); i < e; i++) {
    auto res = op->getResult(i);
    res_values.push_back(AddValue(res));
  }
  impl_->cur_op->SetResults(res_values);

  // process attribute
  auto& table = function_table ? *function_table : impl_->func_defs;
  {
    // lookup the callee function
    auto it = table.find(callee_name.getValue().str());
    CHECK(it != table.end()) << "can't find function ["
                             << callee_name.getValue().str() << "]";
    auto* function =
        impl_->cur_op->CreateFunctionExecutable(it->second, &impl_->func_defs);
    impl_->cur_op->AppendAttribute(new Value(function));
  }

  VLOG(3) << "Emit call " << callee_name.getValue().str() << " "
          << impl_->cur_op->frame();
  return true;
}

MlirToRuntimeTranslator::MlirToRuntimeTranslator(CoreRuntimeBuilder* runtime)
    : impl_(new Impl) {
  CHECK(runtime);
  impl_->runtime = runtime;
}

Value* MlirToRuntimeTranslator::AddValue(mlir::Value mlir_value, Value* value) {
  auto it = impl_->value_map.try_emplace(mlir_value, ValueRef(value));
  CHECK(it.second) << "duplicate add value " << DumpToString(mlir_value);
  return value;
}

void MlirToRuntimeTranslate(mlir::ModuleOp module,
                            CoreRuntimeBuilder* runtime) {
  MlirToRuntimeTranslator(module, runtime).Run();
}

/**
 * Execute the mlir program in test mode -- print some debug information to
 * stdout.
 */
class MlirProgramTestExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder core_runtime;

  MlirProgramTestExecutor(mlir::ModuleOp module, KernelRegistry* registry)
      : MlirToRuntimeTranslator(module, &core_runtime),
        core_runtime(registry),
        registry(registry) {
    CHECK(registry);
  }

  void Run() {
    EmitFunctions();

    CHECK(registry);
    for (auto func_op : impl_->module.getOps<mlir::FuncOp>()) {
      VLOG(3) << "Running function " << func_op.getName().str();
      EmitAndRunFuncWithoutArguments(func_op);
    }
  }

 protected:
  std::unordered_map<std::string, mlir::FuncOp> func_def_table;

  void EmitFunction(mlir::FuncOp op) override {
    CHECK(!impl_->func_defs.count(op.getName().str()))
        << "Duplicate function defition found for function ["
        << op.getName().str();
    impl_->func_defs.emplace(op.getName().str(), op);
  }

 private:
  void EmitAndRunFuncWithoutArguments(mlir::FuncOp func) {
    // print the function name for llvm FileChecker macro, CHECK-LABEL
    std::cout << '@' << func.getName().str() << std::endl;
    if (func.getNumArguments() ==
        0) {  // an entry function, execute it immediately
      VLOG(3) << "executing function " << func.getName().str();
      // Emit and execute each function
      CoreRuntimeBuilder runtime(registry);
      impl_->runtime = &runtime;

      auto& blocks = func.getBlocks();
      CHECK_EQ(blocks.size(), 1UL)
          << "function with more than one block is not supported yet";

      for (auto& op : blocks.front()) {
        if (EmitConstantOp(&op)) continue;
        if (EmitBuildShapeOp(&op)) continue;
        llvm::SmallVector<mlir::Value, 3> results;
        if (EmitReturnOp(&op, &results)) continue;
        if (EmitCallOp(&op, &impl_->func_defs)) continue;
        if (EmitGeneralOp(&op)) continue;
        LOG(FATAL) << "Not supported op: " << DumpToString(op);
      }

      runtime.Execute();

    } else {
      VLOG(2) << "get an callable function: " << func.getName().str();
    }
  }

 private:
  KernelRegistry* registry{};
};

void TestMlir(mlir::ModuleOp module, KernelRegistry* registry) {
  MlirProgramTestExecutor execute(module, registry);
  execute.Run();
}

}  // namespace infrt::host_context
