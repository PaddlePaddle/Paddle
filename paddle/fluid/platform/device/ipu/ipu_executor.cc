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

#include "paddle/fluid/platform/device/ipu/ipu_executor.h"

using float16 = paddle::platform::float16;

namespace paddle {
namespace platform {
namespace ipu {

Executor::~Executor() {
  Detach();
  session_.reset();
  executor_resources_.reset();
}

void Executor::Prepare(const std::string &proto) {
  VLOG(10) << "enter Executor::Prepare";

  AcquireDevice();
  executor_resources_ = std::make_unique<ExecutorResources>();

  auto art = popart::AnchorReturnType("All");
  std::map<popart::TensorId, popart::AnchorReturnType> anchor_ids;
  for (const auto &id : compiler_resources_->outputs) {
    anchor_ids.emplace(id, art);
  }
  auto dataFlow = popart::DataFlow(ipu_strategy_->batches_per_step, anchor_ids);

  if (ipu_strategy_->is_training) {
    VLOG(10) << "Creating TrainingSession from Onnx Model...";
    auto optimizer = compiler_resources_->NewOptimizer();
    session_ = popart::TrainingSession::createFromOnnxModel(
        proto, dataFlow, compiler_resources_->loss_var, *optimizer, device_,
        popart::InputShapeInfo(), ipu_strategy_->popart_options,
        ipu_strategy_->popart_patterns);
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto, dataFlow, device_, popart::InputShapeInfo(),
        ipu_strategy_->popart_options, ipu_strategy_->popart_patterns);
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
  // init run step
  step_ = 0;
}

void Executor::Run(const std::vector<const Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs,
                   const framework::ExecutionContext &ctx) {
  VLOG(10) << "enter Executor::Run";
  // inputs
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, PaddleIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = compiler_resources_->inputs[i];
    input_wrappers.emplace(tensor_id, PaddleIArray(inputs[i]));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }
  // anchors
  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, PaddleIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor_id = compiler_resources_->outputs[i];
    // get dims & dtype from session
    auto fetch_info = session_->getInfo(tensor_id);
    auto output_shape = fetch_info.shape();
    if (ipu_strategy_->batches_per_step > 1) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->batches_per_step);
    }
    if (ipu_strategy_->popart_options.enableGradientAccumulation) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->popart_options.accumulationFactor);
    }
    if (ipu_strategy_->popart_options.enableReplicatedGraphs) {
      output_shape.insert(output_shape.begin(),
                          ipu_strategy_->popart_options.replicatedGraphCount);
    }

    auto *tensor = outputs[i];
    tensor->Resize(framework::make_ddim(output_shape));
    auto fetch_dtype = fetch_info.dataType();
    auto paddle_type = PopartType2VarType(fetch_dtype);
    tensor->mutable_data(ctx.GetPlace(), paddle_type);
    anchor_wrappers.emplace(tensor_id, PaddleIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }
  VLOG(10) << "Prepared inputs/anchors";

  if (ipu_strategy_->is_training && compiler_resources_->with_lr_sched) {
    VLOG(10) << "Update learning_rate";
    auto new_lr =
        GetSingleVarFromScope<float>(scope_, compiler_resources_->lr_var);
    VLOG(10) << "New Lr: " << new_lr;
    auto *optimizer = compiler_resources_->UpdateOptimizer(new_lr);
    auto *session = dynamic_cast<popart::TrainingSession *>(session_.get());
    session->updateOptimizerFromHost(optimizer);
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";

  step_++;
  if (ipu_strategy_->is_training &&
      step_ % ipu_strategy_->save_per_n_step == 0) {
    session_->weightsToHost();
    WeightsToPaddle();
    if (ipu_strategy_->save_onnx_checkpoint) {
      session_->modelToHost("test_last" + std::to_string(step_) + ".onnx");
    }
  }
}

void Executor::AcquireDevice() {
  VLOG(10) << "enter Executor::AcquireDevice";
  if (device_) {
    Detach();
    device_.reset();
  }

  bool use_ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (use_ipu_model) {
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1 "}};
    device_ = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        deviceOpts);
  } else {
    device_ =
        popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
            RequestIpus(ipu_strategy_->num_ipus));
    PADDLE_ENFORCE_NOT_NULL(device_, platform::errors::Unavailable(
                                         "Can't attach IPU, ipu_num = %d.",
                                         RequestIpus(ipu_strategy_->num_ipus)));
  }
  VLOG(10) << "leave Executor::AcquireDevice";
}

void Executor::Detach() {
  if (device_ && device_->isAttached()) {
    VLOG(10) << "trying to detach IPU";
    device_->detach();
    VLOG(10) << " detached IPU";
  }
}

void Executor::SetWeightsIO() {
  auto opt_type = compiler_resources_->optimizer_type;
  VLOG(10) << "SetWeightsIO for " << opt_type;
  auto pre_post_fix = GetOptPrePostfix(opt_type);
  for (const auto &weight_id : compiler_resources_->weights) {
    for (const auto &pair : pre_post_fix) {
      // pair.first : popart prefix, pair.second : paddle postfix
      auto popart_var_name = pair.first + weight_id;
      auto paddle_var_name = weight_id + pair.second;

      if (scope_->FindVar(paddle_var_name) == nullptr) {
        continue;
      }

      if (!session_->hasInfo(popart_var_name)) {
        continue;
      }

      auto var = scope_->GetVar(paddle_var_name);
      auto data_ptr = var->GetMutable<framework::LoDTensor>()->data();

      auto tensor_info = session_->getInfo(popart_var_name);
      executor_resources_->weights_io.insert(popart_var_name,
                                             {data_ptr, tensor_info});
      executor_resources_->weights_and_opt_state.emplace_back(
          std::make_pair(popart_var_name, paddle_var_name));
    }
  }
}

// align_to_popart: align dtype to popart if true, else to paddle
void Executor::ConvertWeights(bool align_to_popart) {
  for (auto weight_pair : executor_resources_->weights_and_opt_state) {
    auto paddle_var = scope_->GetVar(weight_pair.second);
    auto paddle_var_dtype = VarType2PopartType(
        paddle_var->GetMutable<framework::LoDTensor>()->type());

    PADDLE_ENFORCE_EQ((paddle_var_dtype == popart::DataType::FLOAT ||
                       paddle_var_dtype == popart::DataType::FLOAT16),
                      true,
                      platform::errors::InvalidArgument(
                          "Currently, we only support FLOAT16 and FLOAT with "
                          "Paddle, but received type is %s.",
                          paddle_var_dtype));

    popart::TensorInfo info = session_->getInfo(weight_pair.first);
    auto popart_var_dtype = info.dataType();
    PADDLE_ENFORCE_EQ((popart_var_dtype == popart::DataType::FLOAT ||
                       popart_var_dtype == popart::DataType::FLOAT16),
                      true,
                      platform::errors::InvalidArgument(
                          "Currently, we only support FLOAT16 and FLOAT with "
                          "popart, but received type is %s.",
                          popart_var_dtype));

    if (paddle_var_dtype == popart_var_dtype) {
      VLOG(10) << weight_pair.first << " and " << weight_pair.second
               << " have the same dtype : " << popart_var_dtype;
      continue;
    } else if (paddle_var_dtype == popart::DataType::FLOAT) {
      VLOG(10) << weight_pair.first << " and " << weight_pair.second
               << " have different dtype : " << popart_var_dtype;
      auto *data_ptr =
          paddle_var->GetMutable<framework::LoDTensor>()->data<float>();

      auto num_elem = info.nelms();
      if (align_to_popart) {
        std::vector<uint16_t> fp16_data;
        std::transform(data_ptr, data_ptr + num_elem,
                       std::back_inserter(fp16_data),
                       [&](float elem) { return popart::floatToHalf(elem); });
        memcpy(reinterpret_cast<void *>(data_ptr), fp16_data.data(),
               num_elem * sizeof(float16));
      } else {
        std::vector<float> fp32_data;
        auto fp16_data_ptr = reinterpret_cast<uint16_t *>(data_ptr);
        std::transform(fp16_data_ptr, fp16_data_ptr + num_elem,
                       std::back_inserter(fp32_data), [&](uint16_t elem) {
                         return popart::halfToFloat(elem);
                       });
        memcpy(reinterpret_cast<void *>(data_ptr), fp32_data.data(),
               num_elem * sizeof(float));
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Convert Paddle FLOAT16 to popart FLOAT"));
    }
  }
}

// |-----------------------------------------------------|
// | Paddle  | Popart  |             Method              |
// |-----------------------------------------------------|
// |  FLOAT  |  FLOAT  |         Paddle -> Popart        |
// |  FLOAT  | FLOAT16 | floatToHalf -> Paddle -> Popart |
// | FLOAT16 |  FLOAT  |         Unimplemented           |
// | FLOAT16 | FLOAT16 |         Paddle -> Popart        |
// |-----------------------------------------------------|
// floatToHalf -> Paddle: cast then save to paddle
// Paddle -> Popart: copy from paddle to popart
void Executor::WeightsFromPaddle() {
  ConvertWeights(true);
  session_->writeWeights(executor_resources_->weights_io);
}

// |-----------------------------------------------------|
// | Paddle  | Popart  |             Method              |
// |-----------------------------------------------------|
// |  FLOAT  |  FLOAT  |         Popart -> Paddle        |
// |  FLOAT  | FLOAT16 | Popart -> Paddle -> halfToFloat |
// | FLOAT16 |  FLOAT  |         Unimplemented           |
// | FLOAT16 | FLOAT16 |         Popart -> Paddle        |
// |-----------------------------------------------------|
// Paddle -> halfToFloat: cast then save to paddle
// Popart -> Paddle: copy from paddle to popart
void Executor::WeightsToPaddle() {
  session_->readWeights(executor_resources_->weights_io);
  ConvertWeights(false);
}

void Executor::SaveModelToHost(const std::string &path) {
  if (session_) {
    session_->weightsToHost();
    WeightsToPaddle();
    session_->modelToHost(path);
  } else {
    LOG(WARNING) << "Model is empty";
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
