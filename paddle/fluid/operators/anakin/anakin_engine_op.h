/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_CUDA

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/engine.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace operators {

using inference::Singleton;
using inference::anakin::AnakinEngine;

class AnakinEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  std::string engine_key_;
  std::string engine_serialized_data_;
  bool use_gpu_;
  bool enable_int8_;

 public:
  AnakinEngineOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    engine_key_ = Attr<std::string>("engine_key");
    auto params = Attr<std::vector<std::string>>("parameters");
    use_gpu_ = Attr<bool>("use_gpu");
    enable_int8_ = Attr<bool>("enable_int8");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    RunAnakin(scope, dev_place);
  }

  void RunAnakin(const framework::Scope &scope,
                 const platform::Place &dev_place) const {
    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    std::map<std::string, framework::LoDTensor *> inputs;
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);

      inputs.insert({x, &t});
    }

    std::map<std::string, framework::LoDTensor *> outputs;
    int output_index = 0;
    for (const auto &y : Outputs("Ys")) {
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();
      outputs.insert({output_maps[output_index], fluid_t});
      output_index += 1;
    }
    if (enable_int8_) {
      Execute<::anakin::Precision::INT8>(inputs, outputs, dev_place);
    } else {
      Execute<::anakin::Precision::FP32>(inputs, outputs, dev_place);
    }
  }

  template <::anakin::Precision PrecisionT>
  void Execute(const std::map<std::string, framework::LoDTensor *> &inputs,
               const std::map<std::string, framework::LoDTensor *> &outputs,
               const platform::Place &dev_place) const {
    if (use_gpu_) {
#ifdef PADDLE_WITH_CUDA
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(dev_place);
      auto stream =
          reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx)
              .stream();
      auto *engine =
          inference::Singleton<inference::anakin::AnakinEngineManager<
              ::anakin::saber::NV, PrecisionT>>::Global()
              .Get(engine_key_);
      engine->Execute(inputs, outputs, stream);
#endif
    } else {
      auto *engine =
          inference::Singleton<inference::anakin::AnakinEngineManager<
              ::anakin::saber::X86, PrecisionT>>::Global()
              .Get(engine_key_);
      engine->Execute(inputs, outputs);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
