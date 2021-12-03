/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/ipu/ipu_executor.h"

namespace paddle {
namespace platform {
namespace ipu {

Executor::Executor() {}

Executor::~Executor() {}

void Executor::Prepare(const std::string &proto,
                       const std::map<std::string, popart::TensorId> &tensors,
                       const std::vector<popart::TensorId> &outputs,
                       std::shared_ptr<popart::DeviceInfo> device) {
  auto art = popart::AnchorReturnType("All");
  std::map<popart::TensorId, popart::AnchorReturnType> anchor_ids;
  for (const auto &id : outputs) {
    anchor_ids.emplace(id, art);
  }

  auto dataFlow = popart::DataFlow(ipu_strategy_->batches_per_step, anchor_ids);

  PADDLE_ENFORCE_NOT_NULL(device, platform::errors::Unavailable(
                                      "IPU device isn't attached, please call "
                                      "IpuBackend::AttachDevice(id) first."));

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Creating TrainingSession from Onnx Model...";
    auto popart_optimizer = GetPopartOptimizer(opt_info);

    auto it = tensors.find(opt_info.GetLoss());
    PADDLE_ENFORCE_NE(
        it, tensors.end(),
        paddle::platform::errors::InvalidArgument(
            "loss_id = %s doesn't exist in popart graph.", opt_info.GetLoss()));

    session_ = popart::TrainingSession::createFromOnnxModel(
        proto, dataFlow, it->second, *popart_optimizer, device,
        popart::InputShapeInfo(), ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto, dataFlow, device, popart::InputShapeInfo(),
        ipu_strategy_->popart_options_,
        popart::Patterns(popart::PatternsLevel::Default));
  }
  VLOG(10) << "Creating session from Onnx Model...done";

  VLOG(10) << "Preparing session device...";
  session_->prepareDevice();
  VLOG(10) << "Preparing session device...done";

  SetWeightsIO();

  VLOG(10) << "Copy weights from paddle to popart...";
  WeightsFromPaddle();
  VLOG(10) << "Copy weights from paddle to popart...done";

  VLOG(10) << "Copy weights from host to device...";
  session_->weightsFromHost();
  VLOG(10) << "Copy weights from host to device...done";

  if (ipu_strategy_->save_init_onnx) {
    session_->modelToHost("test_init.onnx");
  }
}

void Executor::Run(const std::vector<popart::TensorId> &inputs_id,
                   const std::vector<const framework::Tensor *> &inputs,
                   const std::vector<popart::TensorId> &outputs_id,
                   const std::vector<framework::Tensor *> &outputs,
                   const framework::ExecutionContext &ctx) {
  // inputs
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, PaddleIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs_id[i];
    framework::Tensor *tensor = nullptr;
    tensor->ShareDataWith(*inputs[i]);
    input_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }
  // anchors
  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, PaddleIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = outputs_id[i];
    framework::Tensor *tensor = nullptr;
    tensor->ShareDataWith(*outputs[i]);
    // get dims & dtype from session
    auto fetch_info = session_->getInfo(tensor_id);
    auto output_shape = fetch_info.shape();
    if (ipu_strategy_->batches_per_step > 1) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->batches_per_step);
    }
    tensor->Resize(framework::make_ddim(output_shape));
    auto fetch_dtype = fetch_info.dataType();
    auto paddle_type = PopartType2VarType(fetch_dtype);
    tensor->mutable_data(ctx.GetPlace(), paddle_type);
    anchor_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    VLOG(10) << "Update optimizer learning rate...";
    SetLR(GetLRFromScope());
    auto popart_optimizer = GetPopartOptimizer(opt_info);
    auto &session = dynamic_cast<popart::TrainingSession &>(*session_);
    session.updateOptimizerFromHost(popart_optimizer.get());
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";

  if (ipu_strategy_ != nullptr && ipu_strategy_->is_training) {
    session_->weightsToHost();
    WeightsToPaddle();
    if (ipu_strategy_->save_last_onnx) {
      session_->modelToHost("test_last.onnx");
    }
  }
}

void Executor::SetOptimizerType(const std::string &type) {
  opt_info.SetType(type);
}

void Executor::SetLR(float lr_rate) { opt_info.SetLR(lr_rate); }

void Executor::SetOptimizerAttr(const std::string &attr, float value) {
  opt_info.SetAttr(attr, value);
}

void Executor::SetLoss(const std::string &loss) { opt_info.SetLoss(loss); }

void Executor::SetLRVarName(const std::string &name) {
  opt_info.SetLRVarName(name);
}

void Executor::SetWeights(const std::vector<popart::TensorId> &weights) {
  weights_ = weights;
}

void Executor::SetWeightsIO() {
  auto opt_type = opt_info.GetType();
  auto pre_post_fix = GetOptPrePostfix(opt_type);
  for (const auto &weight_id : weights_) {
    for (const auto &pair : pre_post_fix) {
      if (!IsOptimizerSupported(opt_type)) {
        continue;
      }

      // pair.first : popart prefix, pair.second : paddle postfix
      auto popart_var_name = pair.first + weight_id;
      auto paddle_var_name = weight_id + pair.second;

      if (scope_->FindVar(paddle_var_name) == nullptr) {
        continue;
      }

      auto var = scope_->GetVar(paddle_var_name);
      auto data_ptr = var->GetMutable<framework::LoDTensor>()->data<float>();

      auto tensor_info = session_->getInfo(popart_var_name);
      weights_io_.insert(popart_var_name, {data_ptr, tensor_info});
    }
  }
}

void Executor::WeightsFromPaddle() { session_->writeWeights(weights_io_); }

void Executor::WeightsToPaddle() { session_->readWeights(weights_io_); }

void Executor::SetIpuStrategy(const IpuStrategy &strategy) {
  ipu_strategy_ = &strategy;
}

float Executor::GetLRFromScope() {
  auto lr_var = scope_->GetVar(opt_info.GetLRVarName());
  auto tensor = lr_var->Get<framework::LoDTensor>();

  PADDLE_ENFORCE_EQ(tensor.type(), framework::proto::VarType::FP32,
                    platform::errors::InvalidArgument(
                        "LR requiree float, but got (%s).", tensor.type()));

  return tensor.data<float>()[0];
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
