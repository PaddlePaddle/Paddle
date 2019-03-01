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

#ifdef PADDLE_WITH_CUDA

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
  mutable std::unique_ptr<AnakinNvEngineT> anakin_engine_;
  std::string engine_key_;

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
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    RunAnakin(scope, dev_place);
  }

  void RunAnakin(const framework::Scope &scope,
                 const platform::Place &dev_place) const {
    if (anakin_engine_.get() == nullptr) {
      anakin_engine_.reset(new AnakinEngine<NV, Precision::FP32>(true));
      Prepare(scope, dev_place, anakin_engine_.get());
    }

    auto *engine = anakin_engine_.get();
    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    std::map<std::string, framework::LoDTensor *> inputs;
    // Convert input tensor from fluid to engine.
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      auto t_shape = framework::vectorize(t.dims());
      inputs.insert({x, &t});
    }

    std::map<std::string, framework::LoDTensor *> outputs;
    int output_index = 0;
    for (const auto &y : Outputs("Ys")) {
      std::vector<int> ddim =
          engine->Net()->get_out(output_maps[output_index])->valid_shape();
      // we need get the output anakin output shape.
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();
      fluid_t->Resize(framework::make_ddim(ddim));
      fluid_t->mutable_data<float>(boost::get<platform::CUDAPlace>(dev_place));
      outputs.insert({output_maps[output_index], fluid_t});
      output_index += 1;
    }
    engine->Execute(inputs, outputs);
  }

  void Prepare(const framework::Scope &scope, const platform::Place &dev_place,
               AnakinNvEngineT *engine) const {
    LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
                 "kernel etc). This process may cost a lot of time.";
    framework::proto::BlockDesc block_desc;
    block_desc.ParseFromString(Attr<std::string>("subgraph"));

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    inference::Singleton<inference::anakin::AnakinOpConverter>::Global()
        .ConvertBlock(block_desc, param_names_, scope, engine);
    engine->Freeze();
    engine->Optimize();
    engine->Save("anakin.saved");

    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      auto t_shape = framework::vectorize2int(t.dims());
      std::cout << "!!!!!!!!" << t_shape.size() << std::endl;
      std::cout << "!!!!!!!!" << t_shape[0] << " " << t_shape[1] << " "
                << t_shape[2] << " " << t_shape[3] << std::endl;
      std::cout << x << std::endl;
      engine->SetInputShape(x, t_shape);
    }
    engine->InitGraph();
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
