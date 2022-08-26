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

#include <popart/devicemanager.hpp>
#include <popdist/popdist_poplar.hpp>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"

namespace paddle {
namespace platform {
namespace ipu {

namespace {

// Get paddle prefix and popart postfix of weight states
// Format: {popart_postfix, paddle_prefix}
std::vector<std::pair<std::string, std::string>> GetOptPrePostfix(
    const std::string &opt_type) {
  std::vector<std::pair<std::string, std::string>> pre_post_fix;
  // Weight self
  pre_post_fix.push_back(std::make_pair("", ""));

  // Weight states
  // TODO(alleng) support pair("Accl1___", "_moment1_{id!=0}")
  if (opt_type == "adam" || opt_type == "lamb" || opt_type == "adamw") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment1_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_moment2_0"));
    pre_post_fix.push_back(std::make_pair("Step___", "_beta1_pow_acc_0"));
  } else if (opt_type == "momentum") {
    pre_post_fix.push_back(std::make_pair("Accl___", "_velocity_0"));
  } else if (opt_type == "adamax") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_inf_norm__0"));
    pre_post_fix.push_back(std::make_pair("Step___", "_beta1_pow_acc_0"));
  } else if (opt_type == "adagrad") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_moment_0"));
  } else if (opt_type == "adadelta") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "__avg_squared_grad_0"));
    pre_post_fix.push_back(
        std::make_pair("Accl2___", "__avg_squared_update_0"));
  } else if (opt_type == "rmsprop") {
    pre_post_fix.push_back(std::make_pair("Accl1___", "_mean_square_0"));
    pre_post_fix.push_back(std::make_pair("Accl2___", "_mean_grad_0"));
    pre_post_fix.push_back(std::make_pair("Accl3___", "_momentum__0"));
  }
  return pre_post_fix;
}

class PdIArray final : public popart::IArray {
 public:
  explicit PdIArray(const Tensor *tensor) {
    tensor_.ShareDataWith(*tensor);
    for (int i = 0; i < tensor->dims().size(); ++i) {
      shape_.push_back(tensor->dims().at(i));
    }
  }

 public:
  void *data() { return tensor_.data(); }
  popart::DataType dataType() const {
    return PhiDType2PopartDType(tensor_.dtype());
  }
  std::size_t rank() const { return tensor_.dims().size(); }
  int64_t dim(size_t index) const { return tensor_.dims().at(index); }
  std::size_t nelms() const {
    return std::accumulate(shape_.begin(),
                           shape_.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }
  const popart::Shape shape() const { return shape_; }

 private:
  Tensor tensor_;
  std::vector<int64_t> shape_;
};

}  // namespace

Executor::~Executor() { Reset(); }

void Executor::Prepare(const std::string &proto) {
  VLOG(10) << "enter Executor::Prepare";
  compile_only_ = GetBoolEnv("IPU_COMPILE_ONLY");

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
        proto,
        dataFlow,
        compiler_resources_->loss_var,
        *optimizer,
        device_,
        popart::InputShapeInfo(),
        ipu_strategy_->popart_options,
        ipu_strategy_->popart_patterns);
  } else {
    VLOG(10) << "Creating InferenceSession from Onnx Model...";
    session_ = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device_,
        popart::InputShapeInfo(),
        ipu_strategy_->popart_options,
        ipu_strategy_->popart_patterns);
  }
  VLOG(10) << "Creating session from Onnx Model...done";

  if (compile_only_) {
    LOG(INFO)
        << "Save the offline cache as offline_cache.popart in current path.";
    VLOG(10) << "Compile only...";
    session_->compileAndExport("./offline_cache.popart");
    VLOG(10) << "Compile only...done";
    return;
  } else {
    VLOG(10) << "Preparing session device...";
    session_->prepareDevice();
    VLOG(10) << "Preparing session device...done";
  }

  SetWeightsIO();

  VLOG(10) << "Copy weights from paddle to popart...";
  WeightsFromPaddle();
  VLOG(10) << "Copy weights from paddle to popart...done";

  if (ipu_strategy_->random_seed != std::numeric_limits<std::uint64_t>::max()) {
    VLOG(10) << "Setting random seed to: " << ipu_strategy_->random_seed;
    session_->setRandomSeed(ipu_strategy_->random_seed);
  }
}

void Executor::Run(const std::vector<const Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs,
                   const framework::ExecutionContext &ctx) {
  if (compile_only_) {
    LOG(INFO) << "If IPU_COMPILE_ONLY=True, skip exe.run";
    return;
  }

  VLOG(10) << "enter Executor::Run";
  // inputs
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, PdIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = compiler_resources_->inputs[i];
    input_wrappers.emplace(tensor_id, PdIArray(inputs[i]));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }
  // anchors
  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, PdIArray> anchor_wrappers;
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
    tensor->Resize(phi::make_ddim(output_shape));
    auto fetch_dtype = fetch_info.dataType();
    auto paddle_type = PopartDType2VarType(fetch_dtype);
    tensor->mutable_data(ctx.GetPlace(),
                         framework::TransToPhiDataType(paddle_type));
    anchor_wrappers.emplace(tensor_id, PdIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }
  VLOG(10) << "Prepared inputs/anchors";

  if (ipu_strategy_->is_training && compiler_resources_->with_lr_sched) {
    popart::Optimizer *optimizer;
    if (ipu_strategy_->runtime_options.enable_eval) {
      VLOG(10) << "Switch optimizer to eval mode";
      optimizer = compiler_resources_->eval_optimizer.get();
    } else {
      VLOG(10) << "Update learning_rate";
      float new_lr;
      if (ipu_strategy_->is_dynamic) {
        new_lr = ipu_strategy_->lr;
      } else {
        new_lr =
            GetSingleVarFromScope<float>(scope_, compiler_resources_->lr_var);
      }
      VLOG(10) << "New Lr: " << new_lr;
      optimizer = compiler_resources_->UpdateOptimizer(new_lr);
    }
    auto *session = dynamic_cast<popart::TrainingSession *>(session_.get());
    session->updateOptimizerFromHost(optimizer);
  }

  popart::StepIO stepio(popart_inputs, popart_anchors);
  VLOG(10) << "Running...";
  session_->run(stepio);
  VLOG(10) << "Running...done";
}

void Executor::WeightsToHost() {
  if (ipu_strategy_->is_training && session_) {
    WeightsToPaddle();
  } else {
    LOG(WARNING) << "For a non-trainning graph, cannot sync weights from IPU.";
  }
}

void Executor::AcquireDevice() {
  VLOG(10) << "enter Executor::AcquireDevice";
  if (device_) {
    Detach();
    device_.reset();
  }

  bool use_ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  bool enable_distribution = ipu_strategy_->enable_distribution;
  if (use_ipu_model) {
    VLOG(10) << "Create IPU model device...";
    std::map<std::string, std::string> deviceOpts{
        {
            "numIPUs",
            std::to_string(ipu_strategy_->num_ipus),
        },
        {"tilesPerIPU", std::to_string(ipu_strategy_->tiles_per_ipu)},
        {"ipuVersion", "ipu2"},
    };
    device_ = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        deviceOpts);
    VLOG(10) << "Create IPU model device...done";
  } else if (compile_only_) {
    VLOG(10) << "Create offline device...";
    std::map<std::string, std::string> deviceOpts{
        {
            "numIPUs",
            std::to_string(ipu_strategy_->num_ipus),
        },
        {"tilesPerIPU", std::to_string(ipu_strategy_->tiles_per_ipu)},
        {"ipuVersion", "ipu2"},
    };
    device_ =
        popart::DeviceManager::createDeviceManager().createOfflineIPUDevice(
            deviceOpts);
    VLOG(10) << "Create offline device...done";
  } else if (enable_distribution) {
    VLOG(10) << "Create distribution device...";
    auto ipus_per_replica = ipu_strategy_->num_ipus /
                            ipu_strategy_->popart_options.replicatedGraphCount;
    auto device_id = popdist::getDeviceId(ipus_per_replica);
    device_ = popart::DeviceManager::createDeviceManager().acquireDeviceById(
        device_id);
    PADDLE_ENFORCE_NOT_NULL(
        device_,
        errors::Unavailable("Can't attach IPU in distribution, ipu_num = %d.",
                            RequestIpus(ipu_strategy_->num_ipus)));
    VLOG(10) << "Create distribution device...done";
  } else {
    VLOG(10) << "Create IPU device...";
    device_ =
        popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
            RequestIpus(ipu_strategy_->num_ipus));
    PADDLE_ENFORCE_NOT_NULL(
        device_,
        errors::Unavailable("Can't attach IPU, ipu_num = %d.",
                            RequestIpus(ipu_strategy_->num_ipus)));
    VLOG(10) << "Create IPU device...done";
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

void Executor::Reset() {
  Detach();
  session_.reset();
  executor_resources_.reset();
}

void Executor::SetWeightsIO() {
  auto opt_type = compiler_resources_->optimizer_type;
  VLOG(10) << "SetWeightsIO for " << opt_type;
  auto pre_post_fix = GetOptPrePostfix(opt_type);
  for (const auto &weight_pd : compiler_resources_->weights) {
    for (const auto &pair : pre_post_fix) {
      // pair.first : popart prefix, pair.second : paddle postfix
      auto weight_pop = compiler_resources_->tensors[weight_pd];
      auto popart_var = pair.first + weight_pop;
      auto paddle_var = weight_pd + pair.second;

      if (scope_->FindVar(paddle_var) == nullptr) {
        continue;
      }
      if (!session_->hasInfo(popart_var)) {
        continue;
      }

      VLOG(10) << "Connect paddle weight: " << paddle_var
               << " with popart weight: " << popart_var;
      auto var = scope_->GetVar(paddle_var);
      auto data_ptr = var->GetMutable<framework::LoDTensor>()->data();
      auto tensor_info = session_->getInfo(popart_var);
      executor_resources_->weights_io.insert(popart_var,
                                             {data_ptr, tensor_info});
      executor_resources_->weights_and_opt_state.emplace_back(
          std::make_pair(popart_var, paddle_var));
    }
  }
}

// align_to_popart: align dtype to popart if true, else to paddle
void Executor::ConvertWeights(bool align_to_popart) {
  for (auto weight_pair : executor_resources_->weights_and_opt_state) {
    auto paddle_var = scope_->GetVar(weight_pair.second);
    auto paddle_var_dtype = PhiDType2PopartDType(
        paddle_var->GetMutable<framework::LoDTensor>()->dtype());

    PADDLE_ENFORCE_EQ((paddle_var_dtype == popart::DataType::FLOAT ||
                       paddle_var_dtype == popart::DataType::FLOAT16),
                      true,
                      errors::InvalidArgument(
                          "Currently, we only support FLOAT16 and FLOAT with "
                          "Paddle, but received type is %s.",
                          paddle_var_dtype));

    popart::TensorInfo info = session_->getInfo(weight_pair.first);
    auto popart_var_dtype = info.dataType();
    PADDLE_ENFORCE_EQ((popart_var_dtype == popart::DataType::FLOAT ||
                       popart_var_dtype == popart::DataType::FLOAT16),
                      true,
                      errors::InvalidArgument(
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
        std::transform(data_ptr,
                       data_ptr + num_elem,
                       std::back_inserter(fp16_data),
                       [&](float elem) { return popart::floatToHalf(elem); });
        memcpy(reinterpret_cast<void *>(data_ptr),
               fp16_data.data(),
               num_elem * sizeof(float16));
      } else {
        std::vector<float> fp32_data;
        auto fp16_data_ptr = reinterpret_cast<uint16_t *>(data_ptr);
        std::transform(
            fp16_data_ptr,
            fp16_data_ptr + num_elem,
            std::back_inserter(fp32_data),
            [&](uint16_t elem) { return popart::halfToFloat(elem); });
        memcpy(reinterpret_cast<void *>(data_ptr),
               fp32_data.data(),
               num_elem * sizeof(float));
      }
    } else {
      PADDLE_THROW(
          errors::Unimplemented("Convert Paddle FLOAT16 to popart FLOAT"));
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
  session_->weightsFromHost();
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
  session_->weightsToHost();
  session_->readWeights(executor_resources_->weights_io);
  ConvertWeights(false);
}

void Executor::SaveModelToHost(const std::string &path) {
  if (session_) {
    WeightsToPaddle();
    session_->modelToHost(path);
  } else {
    LOG(WARNING) << "Model is empty";
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
