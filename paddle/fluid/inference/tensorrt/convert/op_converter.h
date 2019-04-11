/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

namespace {  // NOLINT

TRT_DT FluidDataType2TRT(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return TRT_DT::kFLOAT;
    case FluidDT::VarType_Type_INT32:
      return TRT_DT::kINT32;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW("unkown type");
  return TRT_DT::kINT32;
}

nvinfer1::Dims Vec2TRT_Dims(const std::vector<int64_t>& shape) {
  PADDLE_ENFORCE_GT(shape.size(), 1UL,
                    "TensorRT' tensor input requires at least 2 dimensions");
  PADDLE_ENFORCE_LE(shape.size(), 4UL,
                    "TensorRT' tensor input requires at most 4 dimensions");
  PADDLE_ENFORCE(shape.size() == 4UL || shape.size() == 2UL);
  if (shape.size() == 4UL)
    return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
  return nvinfer1::DimsCHW(shape[1], 1, 1);
}

}  // namespace // NOLINT

/*
 * Convert Op from Fluid to TensorRT Engine.
 */
class OpConverter {
 public:
  OpConverter() {}

  // Converter logic for an op.
  virtual void operator()(const framework::proto::OpDesc& op,
                          const framework::Scope& scope,
                          bool test_mode = false) {}

  // Convert a single fluid operator and add the corresponding layer to TRT.
  // test_mode: whether the instance executes in an unit test.
  void ConvertOp(const framework::proto::OpDesc& op,
                 const std::unordered_set<std::string>& parameters,
                 const framework::Scope& scope, TensorRTEngine* engine,
                 bool test_mode = false) {
    framework::OpDesc op_desc(op, nullptr);

    OpConverter* it{nullptr};

    if (op_desc.Type() == "mul") {
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL);
      std::string Y = op_desc.Input("Y")[0];
      if (parameters.count(Y)) {
        it = Registry<OpConverter>::Global().Lookup("fc");
      }
    }
    if (op_desc.Type().find("elementwise") != std::string::npos) {
      static std::unordered_set<std::string> add_tensor_op_set{
          "add", "mul", "sub", "div", "max", "min", "pow"};
      // TODO(xingzhaolong): all mul, sub, div
      // static std::unordered_set<std::string> add_weight_op_set {"add", "mul",
      // "sub", "div"};
      static std::unordered_set<std::string> add_weight_op_set{"add", "mul"};
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL);
      int op_type_len = op_desc.Type().size();
      std::string op_type = op_desc.Type().substr(op_type_len - 3, op_type_len);
      std::string Y = op_desc.Input("Y")[0];
      if (parameters.count(Y)) {
        PADDLE_ENFORCE(add_weight_op_set.count(op_type) > 0,
                       "Unsupported elementwise type" + op_type);
        it = Registry<OpConverter>::Global().Lookup("elementwise_" + op_type +
                                                    "_weight");
        PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]",
                                op_desc.Type());
      } else {
        PADDLE_ENFORCE(add_tensor_op_set.count(op_type) > 0,
                       "Unsupported elementwise type" + op_type);
        it = Registry<OpConverter>::Global().Lookup("elementwise_" + op_type +
                                                    "_tensor");
      }
      PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]",
                              op_desc.Type());
    }

    if (op_desc.Type() == "depthwise_conv2d") {
      it = Registry<OpConverter>::Global().Lookup("conv2d");
      PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]",
                              op_desc.Type());
    }

    if (!it) {
      it = Registry<OpConverter>::Global().Lookup(op_desc.Type());
    }
    PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]",
                            op_desc.Type());
    it->SetEngine(engine);
    (*it)(op, scope, test_mode);
  }

  // Convert a fluid block to tensorrt network, NOTE it just convert operators,
  // the INetwork's inputs and outputs should specified in some other modules.
  void ConvertBlock(const framework::proto::BlockDesc& block,
                    const std::unordered_set<std::string>& parameters,
                    const framework::Scope& scope, TensorRTEngine* engine) {
    std::unique_lock<std::mutex> lk(mut_);
    for (int i = 0; i < block.ops_size(); i++) {
      const auto& op = block.ops(i);
      ConvertOp(op, parameters, scope, engine);
    }
  }

  // The scope  here should be inited with the parameter vars.
  void ConvertBlockToTRTEngine(
      framework::BlockDesc* block_desc, const framework::Scope& scope,
      const std::vector<std::string>& inputs,
      const std::unordered_set<std::string>& parameters,
      const std::vector<std::string>& outputs, TensorRTEngine* engine) {
    engine->InitNetwork();
    for (auto& input : inputs) {
      if (parameters.count(input)) continue;
      auto* var = block_desc->FindVar(input);
      PADDLE_ENFORCE(var, "no variable called %s", input);
      PADDLE_ENFORCE_EQ(var->GetType(), FluidDT::VarType_Type_LOD_TENSOR,
                        "TensorRT engine only takes LoDTensor as input");
      auto var_shape = var->GetShape();

      engine->DeclareInput(
          input, FluidDataType2TRT(
                     var->Proto()->type().lod_tensor().tensor().data_type()),
          Vec2TRT_Dims(var_shape));
    }
    framework::proto::BlockDesc* block_proto = block_desc->Proto();
    ConvertBlock(*block_proto, parameters, scope, engine);
    for (auto& output : outputs) {
      engine->DeclareOutput(output);
    }
    engine->FreezeNetwork();
  }

  void SetEngine(TensorRTEngine* engine) { engine_ = engine; }

  virtual ~OpConverter() {}

  // TensorRT engine
  TensorRTEngine* engine_{nullptr};

 protected:
  bool test_mode_;

 private:
  // registered op converter map, whose key is the fluid op type, and value is
  // the pointer position of corresponding OpConverter class.
  std::unordered_map<std::string, OpConverter*> converters_;
  // fluid inference scope
  framework::Scope* scope_{nullptr};
  std::mutex mut_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

#define REGISTER_TRT_OP_CONVERTER(op_type__, Converter__)                      \
  struct trt_##op_type__##_converter : public ::paddle::framework::Registrar { \
    trt_##op_type__##_converter() {                                            \
      ::paddle::inference::Registry<                                           \
          paddle::inference::tensorrt::OpConverter>::Global()                  \
          .Register<::paddle::inference::tensorrt::Converter__>(#op_type__);   \
    }                                                                          \
  };                                                                           \
  trt_##op_type__##_converter trt_##op_type__##_converter__;                   \
  int TouchConverterRegister_##op_type__() {                                   \
    trt_##op_type__##_converter__.Touch();                                     \
    return 0;                                                                  \
  }

#define USE_TRT_CONVERTER(op_type__)                                    \
  extern int TouchConverterRegister_##op_type__();                      \
  static int use_op_converter_trt_##op_type__ __attribute__((unused)) = \
      TouchConverterRegister_##op_type__();
