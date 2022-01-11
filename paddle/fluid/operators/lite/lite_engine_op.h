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
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/lite/tensor_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace operators {

class LiteEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;
  paddle::lite_api::PaddlePredictor *engine_;
  framework::proto::VarType::Type precision_;
  bool use_gpu_;
  bool zero_copy_;

 public:
  LiteEngineOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    in_names_ = Inputs("Xs");
    out_names_ = Outputs("Ys");
    engine_ =
        inference::Singleton<inference::lite::EngineManager>::Global().Get(
            Attr<std::string>("engine_key"));
    if (Attr<bool>("enable_int8")) {
      precision_ = framework::proto::VarType_Type_INT8;
    } else {
      precision_ = framework::proto::VarType_Type_FP32;
    }
    use_gpu_ = Attr<bool>("use_gpu");
    zero_copy_ = Attr<bool>("zero_copy");
  }

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    Execute(scope, dev_place);
  }

  void Execute(const framework::Scope &scope,
               const platform::Place &dev_place) const {
    const platform::DeviceContext *ctx =
        platform::DeviceContextPool::Instance().Get(dev_place);
    for (size_t i = 0; i < in_names_.size(); i++) {
      framework::LoDTensor src_t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope,
                                                                  in_names_[i]);
      paddle::lite_api::Tensor dst_t = *(engine_->GetInput(i));
      VLOG(3) << "== fluid -> lite (" << in_names_[i] << " -> "
              << engine_->GetInputNames()[i] << ")";
      inference::lite::utils::TensorCopy(&dst_t, &src_t, *ctx, zero_copy_);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(dev_place)) {
      platform::GpuStreamSync(
          static_cast<const platform::CUDADeviceContext *>(ctx)->stream());
    }
#endif
    VLOG(3) << "lite engine run";
    engine_->Run();
    VLOG(3) << "lite engine run done";
    for (size_t i = 0; i < out_names_.size(); i++) {
      paddle::lite_api::Tensor src_t = *(engine_->GetOutput(i));
      framework::LoDTensor *dst_t =
          &inference::analysis::GetFromScope<framework::LoDTensor>(
              scope, out_names_[i]);
      VLOG(3) << "== lite -> fluid (" << out_names_[i] << " -> "
              << engine_->GetOutputNames()[i] << ")";
      inference::lite::utils::TensorCopy(dst_t, &src_t, *ctx, zero_copy_);
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(dev_place)) {
      platform::GpuStreamSync(
          static_cast<const platform::CUDADeviceContext *>(ctx)->stream());
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle
