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
#ifndef PADDLE_INFRT_HOST_CONTEXT_PADDLE_MLIR_H_
#define PADDLE_INFRT_HOST_CONTEXT_PADDLE_MLIR_H_

#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <fstream>
#include <iostream>
#include <string>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/common/string.h"
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/ir/basic_kernels.h"
#include "paddle/infrt/dialect/init_dialects.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/tensor_shape.h"
#include "paddle/infrt/paddle/model_parser.h"

class MLIRModelGenImpl {
 public:
  MLIRModelGenImpl();
  mlir::ModuleOp ImportPaddleModel(const std::string &model_file,
                                   const std::string &param_file);
  mlir::ModuleOp ImportPaddleModel(const std::string &model_dir);

 private:
  // parse paddle model file
  infrt::paddle::framework_proto::ProgramDesc ParsePaddleModel(
      const std::string &model_file);

  // convert paddle model proto into paddle dialect module
  mlir::ModuleOp ImportPaddleModel(
      const infrt::paddle::framework_proto::ProgramDesc &program);

  // get inputs and outputs info from program_desc
  llvm::SmallVector<mlir::Type, 4> GetModelInputsType(
      const infrt::paddle::framework_proto::ProgramDesc &program);
  llvm::SmallVector<mlir::Type, 4> GetModelOutputsType(
      const infrt::paddle::framework_proto::ProgramDesc &program);
  // create main function module
  mlir::FuncOp UpdateModelModule(llvm::SmallVector<mlir::Type, 4> operandTypes,
                                 llvm::SmallVector<mlir::Type, 4> resultTypes);
  // convert paddle ops into paddle dialect ops (in mlir form)
  void UpdateModelOps(
      const infrt::paddle::framework_proto::ProgramDesc &program);
  // convert persistable params and inputs variable into mlir domain
  void UpdateModelParams(
      const infrt::paddle::framework_proto::ProgramDesc &program,
      mlir::FuncOp *mainFunc);
  // register model outpus into params_map_
  void UpdateModelOutputs(
      const infrt::paddle::framework_proto::ProgramDesc &program);

  // method for converting proto::op into op in paddle dialect
  void buildOperation(const infrt::paddle::framework_proto::OpDesc &op_);

  llvm::SmallVector<mlir::Value, 4> GetOpInputValue(
      const infrt::paddle::framework_proto::OpDesc &op_);
  llvm::SmallVector<mlir::Type, 4> GetOpOutputType(
      const infrt::paddle::framework_proto::OpDesc &op_);
  llvm::SmallVector<mlir::NamedAttribute, 4> GetOpAttributes(
      const infrt::paddle::framework_proto::OpDesc &op_);
  void RegisterOpOutputVars(const infrt::paddle::framework_proto::OpDesc &op_,
                            mlir::Operation *mlir_op_);

  mlir::MLIRContext *context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  infrt::paddle::framework_proto::BlockDesc main_block_;

  std::map<std::string, mlir::Value> params_map_;
};

// convert protobuf repeated to std::vector.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(
      repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
  return ret;
}
// convert proto type to mlir type
bool ConvertDataType(infrt::paddle::framework_proto::VarType::Type dtype,
                     mlir::Builder builder,
                     mlir::Type *type);
bool ConvertDataTypeToPhi(infrt::paddle::framework_proto::VarType::Type dtype,
                          infrt::PrecisionType *type);

#endif  // PADDLE_INFRT_HOST_CONTEXT_PADDLE_MLIR_H_
