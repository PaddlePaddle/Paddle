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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "framework/core/types.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/anakin/convert/registrar.h"
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
    std::shared_ptr<AnakinOpConverter> it{nullptr};

    if (op_type == "mul") {
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL);
      std::string Y = op_desc.Input("Y")[0];
      std::cout << Y << parameters.count(Y) << std::endl;
      if (parameters.count(Y)) {
        it = OpRegister::instance()->Get("fc");
      }
    }

    if (!it) {
      it = OpRegister::instance()->Get(op_type);
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

#define REGISTER_ANAKIN_OP_CONVERTER(op_type__, Converter__)                \
  struct anakin_##op_type__##_converter                                     \
      : public ::paddle::framework::Registrar {                             \
    anakin_##op_type__##_converter() {                                      \
      ::paddle::inference::                                                 \
          Registry<paddle::inference::anakin::AnakinOpConverter>::Register< \
              ::paddle::inference::anakin::Converter__>(#op_type__);        \
    }                                                                       \
  };                                                                        \
  anakin_##op_type__##_converter anakin_##op_type__##_converter__;          \
  int TouchConverterRegister_anakin_##op_type__() {                         \
    anakin_##op_type__##_converter__.Touch();                               \
    return 0;                                                               \
  }

#define USE_ANAKIN_CONVERTER(op_type__)                                    \
  extern int TouchConverterRegister_anakin_##op_type__();                  \
  static int use_op_converter_anakin_##op_type__ __attribute__((unused)) = \
      TouchConverterRegister_anakin_##op_type__();
