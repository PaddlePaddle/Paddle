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

using FluidDT = framework::proto::VarType_Type;
using inference::Singleton;

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::Precision;
using anakin::saber::NV;
using anakin::saber::X86;
using anakin::saber::Shape;
using anakin::PBlock;
using anakin::PTuple;
using inference::anakin::AnakinEngine;

class AnakinEngineOp : public framework::OperatorBase {
  using AnakinNvEngineT = AnakinEngine<NV, Precision::FP32>;

 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  mutable AnakinNvEngineT *anakin_engine_;
  std::string engine_key_;
  std::string engine_serialized_data_;

 public:
  AnakinEngineOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    engine_key_ = Attr<std::string>("engine_key");
    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
    anakin_engine_ = nullptr;
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    RunAnakin(scope, dev_place);
  }

  void RunAnakin(const framework::Scope &scope,
                 const platform::Place &dev_place) const {
    auto *engine = GetEngine(scope, dev_place);
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx).stream();

    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    std::map<std::string, framework::LoDTensor *> inputs;
    // Convert input tensor from fluid to engine.
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
    engine->Execute(inputs, outputs, stream);
  }

  AnakinNvEngineT *GetEngine(const framework::Scope &scope,
                             const platform::Place &dev_place) const {
    if (anakin_engine_ == nullptr) {
      anakin_engine_ =
          inference::Singleton<inference::anakin::AnakinEngineManager>::Global()
              .Get(engine_key_);
    }

    return anakin_engine_;
  }

  void Prepare(const framework::Scope &scope, const platform::Place &dev_place,
               AnakinNvEngineT *engine) const {
    LOG(INFO) << "Prepare Anakin engine (Optimize model structure, Select OP "
                 "kernel etc). This process may cost a lot of time.";
    framework::proto::BlockDesc block_desc;
    block_desc.ParseFromString(Attr<std::string>("subgraph"));

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    inference::Singleton<inference::anakin::AnakinOpConverter>::Global()
        .ConvertBlock(block_desc, param_names_, scope, engine);
    engine->Freeze();
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      auto t_shape = framework::vectorize2int(t.dims());
      // all input shape should be 4 dims
      if (t_shape.size() == 2) {
        t_shape.push_back(1);
        t_shape.push_back(1);
      }
      engine->SetInputShape(x, t_shape);
    }

    engine->Optimize();

    engine->InitGraph();
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
