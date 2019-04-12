// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "framework/core/types.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/anakin/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "saber/saber_types.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
class AnakinOpConverter {
  using AnakinEngineT = AnakinEngine<TargetT, PrecisionT>;

 public:
  AnakinOpConverter() = default;

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::BlockDesc &block_desc,
                          const framework::Scope &scope, bool test_mode) {}
  void ConvertOp(const framework::proto::OpDesc &op,
                 const framework::BlockDesc &block_desc,
                 const std::unordered_set<std::string> &parameters,
                 const framework::Scope &scope, AnakinEngineT *engine,
                 bool test_mode = false) {
    framework::OpDesc op_desc(op, nullptr);
    std::string op_type = op_desc.Type();
    AnakinOpConverter *it = nullptr;
    if (op_type == "depthwise_conv2d") op_type = "conv2d";
    if (op_type == "reshape2") op_type = "reshape";
    if (op_type == "transpose2") op_type = "transpose";
    if (op_type == "flatten2") op_type = "flatten";

    if (!it) {
      it = Registry<AnakinOpConverter>::Global().Lookup(op_type);
    }
    PADDLE_ENFORCE_NOT_NULL(it, "no OpConverter for optype [%s]", op_type);
    it->SetEngine(engine);
    (*it)(op, block_desc, scope, test_mode);
  }

  void ConvertBlock(framework::BlockDesc *block_desc,
                    const std::unordered_set<std::string> &parameters,
                    const framework::Scope &scope, AnakinEngineT *engine) {
    std::unique_lock<std::mutex> lock(mutex_);
    framework::proto::BlockDesc *block = block_desc->Proto();
    for (auto i = 0; i < block->ops_size(); i++) {
      auto &op = block->ops(i);
      ConvertOp(op, *block_desc, parameters, scope, engine);
    }
  }

  // The scope  here should be inited with the parameter vars.
  void ConvertBlockToAnakinEngine(
      framework::BlockDesc *block_desc, framework::Scope *scope,
      const std::vector<std::string> &inputs,
      const std::unordered_set<std::string> &parameters,
      const std::vector<std::string> &outputs, AnakinEngineT *engine) {
    ConvertBlock(block_desc, parameters, *scope, engine);
    // if the max_batch size
    int max_batch_size = engine->GetMaxBatchSize();
    PADDLE_ENFORCE(max_batch_size > 0,
                   "the max_batch_size setted from config->EnableAnakinEngine "
                   "must largger than 0");
    // If the user does not specify this variable, we use the input shape from
    // the block_desc.
    auto max_input_shape = engine->GetMaxInputShape();
    std::map<std::string, std::vector<int>> temp_max_input_shape;
    // Register outputs with anakin using the RegistVar interface before Freeze.
    // Note that RegistVar's parameters can only be outputs, not inputs.
    for (auto &output : outputs) {
      engine->Graph()->RegistVar(output);
    }
    engine->Freeze();
    // Add scale for tensor in int8 mode.
    auto tensor_scales = engine->GetTensorScales();

    for (auto &item : tensor_scales) {
      engine->Graph()->SetVarScale(item.first, item.second);
    }

    for (auto &input : inputs) {
      if (parameters.count(input)) continue;
      std::vector<int> input_shape;
      input_shape.resize(4);
      input_shape[0] = max_batch_size;
      if (max_input_shape.count(input)) {
        PADDLE_ENFORCE(max_input_shape[input].size() == 4,
                       "the dimensions of max_input_shape setted from "
                       "config->EnableAnakinEngine must be 4");
        for (int i = 1; i < 4; i++) {
          input_shape[i] = max_input_shape[input][i];
        }
      } else {
        auto *var = block_desc->FindVar(input);
        PADDLE_ENFORCE(var, "no variable called %s", input);

        auto var_shape = var->GetShape();
        std::cout << "input :" << input << std::endl;
        PADDLE_ENFORCE(var_shape.size() == 4);

        for (size_t i = 1; i < var_shape.size(); i++) {
          input_shape[i] = var_shape[i];
        }
      }
      temp_max_input_shape[input] = input_shape;
      engine->SetInputShape(input, input_shape);
    }
    engine->SetMaxInputShape(temp_max_input_shape);
    engine->Optimize();
    engine->InitNet();
  }

  void SetEngine(AnakinEngineT *engine) { engine_ = engine; }
  virtual ~AnakinOpConverter() {}

 protected:
  bool test_mode_;
  AnakinEngineT *engine_{nullptr};

 private:
  std::unordered_map<std::string, AnakinOpConverter<TargetT, PrecisionT> *>
      converters_;
  framework::Scope *scope_{nullptr};
  std::mutex mutex_;
};

template class AnakinOpConverter<::anakin::saber::NV,
                                 ::anakin::Precision::FP32>;
template class AnakinOpConverter<::anakin::saber::NV,
                                 ::anakin::Precision::INT8>;

template class AnakinOpConverter<::anakin::saber::X86,
                                 ::anakin::Precision::FP32>;
template class AnakinOpConverter<::anakin::saber::X86,
                                 ::anakin::Precision::INT8>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#define REGISTER_ANAKIN_OP_CONVERTER_BASE(op_type__, Converter__,              \
                                          place_type__, place_class__,         \
                                          precision_type__, precision_class__) \
  struct anakin_##op_type__##_##place_type__##_##precision_type__##_converter  \
      : public ::paddle::framework::Registrar {                                \
    anakin_##op_type__##_##place_type__##_##precision_type__##_converter() {   \
      LOG(INFO) << "register convert " << #op_type__ << " ";                   \
      ::paddle::inference::Registry<                                           \
          ::paddle::inference::anakin::AnakinOpConverter<                      \
              place_class__, precision_class__>>::Global()                     \
          .Register<Converter__>(#op_type__);                                  \
    }                                                                          \
  };                                                                           \
  anakin_##op_type__##_##place_type__##_##precision_type__##_converter         \
      anakin_##op_type__##_##place_type__##_##precision_type__##_converter__;  \
  int Touch_anakin_##op_type__##_##place_type__##_##precision_type__() {       \
    anakin_##op_type__##_##place_type__##_##precision_type__##_converter__     \
        .Touch();                                                              \
    return 0;                                                                  \
  }

#define REGISTER_CUDA_ANAKIN_OP_CONVERTER(op_type__, Converter__) \
  REGISTER_ANAKIN_OP_CONVERTER_BASE(op_type__, Converter__, CUDA, \
                                    ::anakin::saber::NV, FP32,    \
                                    ::anakin::Precision::FP32)

#define REGISTER_CUDA_INT8_ANAKIN_OP_CONVERTER(op_type__, Converter__) \
  REGISTER_ANAKIN_OP_CONVERTER_BASE(op_type__, Converter__, CUDA,      \
                                    ::anakin::saber::NV, INT8,         \
                                    ::anakin::Precision::INT8)

#define REGISTER_CPU_ANAKIN_OP_CONVERTER(op_type__, Converter__) \
  REGISTER_ANAKIN_OP_CONVERTER_BASE(op_type__, Converter__, CPU, \
                                    ::anakin::saber::X86, FP32,  \
                                    ::anakin::Precision::FP32)

#define REGISTER_CPU_INT8_ANAKIN_OP_CONVERTER(op_type__, Converter__) \
  REGISTER_ANAKIN_OP_CONVERTER_BASE(op_type__, Converter__, CPU,      \
                                    ::anakin::saber::X86, INT8,       \
                                    ::anakin::Precision::INT8)

#define USE_ANAKIN_CONVERTER_BASE(op_type__, place_type__, precision_type__)   \
  extern int Touch_anakin_##op_type__##_##place_type__##_##precision_type__(); \
  int use_converter_anakin_##op_type__##_##place_type__##_##precision_type__   \
      __attribute__((unused)) =                                                \
          Touch_anakin_##op_type__##_##place_type__##_##precision_type__();

#define USE_ANAKIN_CONVERTER(op_type__) \
  USE_ANAKIN_CONVERTER_BASE(op_type__, CUDA, FP32)
#define USE_INT8_ANAKIN_CONVERTER(op_type__) \
  USE_ANAKIN_CONVERTER_BASE(op_type__, CUDA, INT8)

#define USE_CPU_ANAKIN_CONVERTER(op_type__) \
  USE_ANAKIN_CONVERTER_BASE(op_type__, CPU, FP32)
#define USE_CPU_INT8_ANAKIN_CONVERTER(op_type__) \
  USE_ANAKIN_CONVERTER_BASE(op_type__, CPU, INT8)
