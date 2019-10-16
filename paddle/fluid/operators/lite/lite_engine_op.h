/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/analysis/helper.h"

#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/lite/tensor_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace operators {

class LiteEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;
  paddle::lite::Predictor* engine_;

 public:
  LiteEngineOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    in_names_ = Inputs("Xs");
    out_names_ = Outputs("Ys");
    engine_ = inference::Singleton<inference::lite::EngineManager>::Global()
            .Get(Attr<std::string>("engine_key"));
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    Execute(scope, dev_place);
  }

  void Execute(const framework::Scope &scope,
               const platform::Place &dev_place) const {
    for (size_t i = 0; i < in_names_.size(); i++) {
      const framework::LoDTensor& src_t = inference::analysis::GetFromScope<framework::LoDTensor>(scope, in_names_[i]);
      paddle::lite::Tensor* dst_t = engine_->GetInput(i);
      inference::lite::TensorCopy(dst_t, src_t);
    }
    engine_->Run();
    cudaDeviceSynchronize();
    for (size_t i = 0; i < out_names_.size(); i++) {
      const paddle::lite::Tensor& src_t = *(engine_->GetOutput(i));
      framework::LoDTensor* dst_t = &inference::analysis::GetFromScope<framework::LoDTensor>(scope, out_names_[i]);
      inference::lite::TensorCopy(dst_t, src_t);
    }
  }
};

}  // namespace operators
}  // namespace paddle
