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

using AnakinNvEngine =
    AnakinEngine<::anakin::saber::NV, ::anakin::Precision::FP32>;

class AnakinOpConverter {
 public:
  AnakinOpConverter() = default;

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::Scope &scope, bool test_mode) {}
  void ConvertOp(const framework::proto::OpDesc &op,
                 const std::unordered_set<std::string> &parameters,
                 const framework::Scope &scope, AnakinNvEngine *engine,
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
    (*it)(op, scope, test_mode);
  }

  void ConvertBlock(const framework::proto::BlockDesc &block,
                    const std::unordered_set<std::string> &parameters,
                    const framework::Scope &scope, AnakinNvEngine *engine) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto i = 0; i < block.ops_size(); i++) {
      auto &op = block.ops(i);
      ConvertOp(op, parameters, scope, engine);
    }
  }

  // The scope  here should be inited with the parameter vars.
  void ConvertBlockToAnakinEngine(
      framework::BlockDesc *block_desc, framework::Scope *scope,
      const std::vector<std::string> &inputs,
      const std::unordered_set<std::string> &parameters,
      const std::vector<std::string> &outputs, AnakinNvEngine *engine) {
    framework::proto::BlockDesc *block_proto = block_desc->Proto();
    ConvertBlock(*block_proto, parameters, *scope, engine);

    engine->Freeze();
    // if the max_batch size
    int max_batch_size = engine->GetMaxBatchSize();
    PADDLE_ENFORCE(max_batch_size > 0,
                   "the max_batch_size setted from config->EnableAnakinEngine "
                   "must largger than 0");
    // If the user does not specify this variable, we use the input shape from
    // the block_desc.
    auto max_input_shape = engine->GetMaxInputShape();
    std::map<std::string, std::vector<int>> temp_max_input_shape;

    for (auto &input : inputs) {
      if (parameters.count(input)) continue;
      std::vector<int> input_shape;
      input_shape.resize(4);
      input_shape[0] = max_batch_size;
      if (max_input_shape.count(input)) {
        PADDLE_ENFORCE(max_input_shape[input].size() == 4,
                       "the dimensions of  max_input_shape setted from "
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
      engine->Graph()->RegistVar(input);  // For share from data.
    }
    engine->SetMaxInputShape(temp_max_input_shape);
    engine->Optimize();

    // For anakin share with fluid tensor.
    engine->AllocTmpMem();
    engine->InitGraph();
  }

  void SetEngine(AnakinNvEngine *engine) { engine_ = engine; }
  virtual ~AnakinOpConverter() {}

 protected:
  bool test_mode_;
  AnakinNvEngine *engine_{nullptr};

 private:
  std::unordered_map<std::string, AnakinOpConverter *> converters_;
  framework::Scope *scope_{nullptr};
  std::mutex mutex_;
};

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#define REGISTER_ANAKIN_OP_CONVERTER(op_type__, Converter__)               \
  struct anakin_##op_type__##_converter                                    \
      : public ::paddle::framework::Registrar {                            \
    anakin_##op_type__##_converter() {                                     \
      LOG(INFO) << "register convert " << #op_type__;                      \
      ::paddle::inference::Registry<                                       \
          ::paddle::inference::anakin::AnakinOpConverter>::Global()        \
          .Register<::paddle::inference::anakin::Converter__>(#op_type__); \
    }                                                                      \
  };                                                                       \
  anakin_##op_type__##_converter anakin_##op_type__##_converter__;         \
  int TouchConverterRegister_anakin_##op_type__() {                        \
    anakin_##op_type__##_converter__.Touch();                              \
    return 0;                                                              \
  }

#define USE_ANAKIN_CONVERTER(op_type__)                             \
  extern int TouchConverterRegister_anakin_##op_type__();           \
  int use_op_converter_anakin_##op_type__ __attribute__((unused)) = \
      TouchConverterRegister_anakin_##op_type__();
