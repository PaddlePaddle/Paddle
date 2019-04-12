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

#include "paddle/fluid/inference/lite/api_anakin_engine.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework/core/net/net.h"
#include "framework/operators/ops.h"
#include "saber/funcs/timer.h"

namespace paddle {

using paddle::contrib::AnakinConfig;

template <typename Target, anakin::Precision Precision>
PaddleInferenceAnakinPredictor<Target, Precision>::
    PaddleInferenceAnakinPredictor(const contrib::AnakinConfig &config) {
  anakin::saber::Env<Target>::env_init();
#ifdef USE_ARM_PLACE
  anakin::saber::Context<Target> ctx;
  // set mode and thread number
  anakin::saber::PowerMode mode = anakin::saber::SABER_POWER_HIGH;
  ctx.set_run_mode(mode, config.thread_num);
// ctx.set_arch(anakin::A73);
// ctx.set_cache(32 * 1024, 512 * 1024, 0);
#endif
  CHECK(Init(config));
}

template <typename Target, anakin::Precision Precision>
bool PaddleInferenceAnakinPredictor<Target, Precision>::Init(
    const contrib::AnakinConfig &config) {
  if (!(graph_.load(config.model_file))) {
    LOG(INFO) << "fail to load graph from " << config.model_file;
    return false;
  }

  auto inputs = graph_.get_ins();
  for (auto &input_str : inputs) {
    graph_.ResetBatchSize(input_str, config.max_batch_size);
    max_batch_size_ = config.max_batch_size;
  }
  // optimization for graph
  if (!(graph_.Optimize())) {
    return false;
  }
  // construct executer
  if (executor_p_ == nullptr) {
    executor_p_ = new anakin::Net<Target, Precision>(graph_, true);
  }
  return true;
}

template <typename Target, anakin::Precision Precision>
bool PaddleInferenceAnakinPredictor<Target, Precision>::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data, int batch_size) {
  for (const auto &input : inputs) {
    if (input.dtype != PaddleDType::FLOAT32) {
      LOG(INFO) << "Only support float type inputs. " << input.name
                << "'s type is not float";
      return false;
    }
    auto d_tensor_in_p = executor_p_->get_in(input.name);
    auto net_shape = d_tensor_in_p->shape();
    if (net_shape.size() != input.shape.size()) {
      LOG(INFO) << " input  " << input.name
                << "'s shape size should be equal to that of net";
      return false;
    }
    int sum = 1;
    for_each(input.shape.begin(), input.shape.end(), [&](int n) { sum *= n; });
    if (sum > net_shape.count()) {
      graph_.Reshape(input.name, input.shape);
      delete executor_p_;
      executor_p_ = new anakin::Net<Target, Precision>(graph_, true);
      d_tensor_in_p = executor_p_->get_in(input.name);
    }

    anakin::saber::Shape tmp_shape;
    for (auto s : input.shape) {
      tmp_shape.push_back(s);
    }
    d_tensor_in_p->reshape(tmp_shape);

    if (input.lod.size() > 0) {
      if (input.lod.size() > 1) {
        LOG(INFO) << " input lod first dim should <=1, but you set "
                  << input.lod.size();
        return false;
      }
      std::vector<int> offset(input.lod[0].begin(), input.lod[0].end());
      d_tensor_in_p->set_seq_offset({offset});
      LOG(INFO) << "offset.size(): " << offset.size();
      for (int i = 0; i < offset.size(); i++) {
        LOG(INFO) << offset[i];
      }
    }

    void *d_data_p = d_tensor_in_p->mutable_data();
    if (std::is_same<anakin::ARM, Target>::value) {
      memcpy(d_data_p, static_cast<float *>(input.data.data()),
             d_tensor_in_p->valid_size() * sizeof(float));
    }
  }

  if (output_data->empty()) {
    LOG(INFO) << "At least one output should be set with tensors' names.";
    return false;
  }
  // run prediction
  executor_p_->prediction();

  for (auto &output : *output_data) {
    auto *tensor = executor_p_->get_out(output.name);
    output.shape = tensor->valid_shape();
    if (output.data.length() < tensor->valid_size() * sizeof(float)) {
      output.data.Resize(tensor->valid_size() * sizeof(float));
    }

    if (std::is_same<anakin::ARM, Target>::value) {
      memcpy(output.data.data(), tensor->mutable_data(),
             tensor->valid_size() * sizeof(float));
    }
  }
  return true;
}

template <typename Target, anakin::Precision Precision>
anakin::Net<Target, Precision>
    &PaddleInferenceAnakinPredictor<Target, Precision>::get_executer() {
  return *executor_p_;
}

// the cloned new Predictor of anakin share the same net weights from original
// Predictor
template <typename Target, anakin::Precision Precision>
std::unique_ptr<PaddlePredictor>
PaddleInferenceAnakinPredictor<Target, Precision>::Clone() {
  LOG(INFO) << "Anakin Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(
      new PaddleInferenceAnakinPredictor<Target, Precision>());
  // construct executer from other graph
  auto anakin_predictor_p =
      dynamic_cast<PaddleInferenceAnakinPredictor<Target, Precision> *>(
          cls.get());
  if (!anakin_predictor_p) {
    LOG(INFO) << "fail to call Init";
    return nullptr;
  }
  anakin_predictor_p->get_executer().init(graph_);

  return std::move(cls);
}

template class PaddleInferenceAnakinPredictor<anakin::ARM,
                                              anakin::Precision::FP32>;
template class PaddleInferenceAnakinPredictor<anakin::ARM,
                                              anakin::Precision::INT8>;

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
    const contrib::AnakinConfig &config) {
  if (config.target_type != contrib::AnakinConfig::ARM) {
    LOG(INFO) << "Anakin Predictor: Only ARM platform is supported currently.";
    return nullptr;
  }

  LOG(INFO) << "Anakin Predictor create.";
  if (config.precision_type == contrib::AnakinConfig::FP32) {
    LOG(INFO) << "Anakin Predictor create on [ FP32 ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::ARM,
                                           anakin::Precision::FP32>(config));
    return x;
  } else if (config.precision_type == contrib::AnakinConfig::INT8) {
    LOG(INFO) << "Anakin Predictor create on [ INT8 ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::ARM,
                                           anakin::Precision::INT8>(config));
    return x;
  } else {
    LOG(INFO) << "Anakin Predictor create on unsupported precision.";
    return nullptr;
  }
}

#ifdef PADDLE_ANAKIN_ENABLE_OP_TIMER
template <typename Target, anakin::Precision Precision>
using executor_t = anakin::Net<Target, Precision>;

template <typename Target, anakin::Precision Precision>
void DisplayOpTimer(executor_t<Target, Precision> *net_executor, int epoch) {
  std::vector<float> op_time = net_executor->get_op_time();
  auto exec_funcs = net_executor->get_exec_funcs();
  auto op_param = net_executor->get_op_param();
  for (int i = 0; i < op_time.size(); i++) {
    LOG(INFO) << "name: " << exec_funcs[i].name
              << " op_type: " << exec_funcs[i].op_name
              << " op_param: " << op_param[i] << " time " << op_time[i] / epoch;
  }
  std::map<std::string, float> op_map;
  for (int i = 0; i < op_time.size(); i++) {
    auto it = op_map.find(op_param[i]);
    if (it != op_map.end())
      op_map[op_param[i]] += op_time[i];
    else
      op_map.insert(std::pair<std::string, float>(op_param[i], op_time[i]));
  }
  for (auto it = op_map.begin(); it != op_map.end(); ++it) {
    LOG(INFO) << it->first << "  " << (it->second) / epoch << " ms";
  }
}
#endif

template <typename Target, anakin::Precision Precision>
PaddleInferenceAnakinPredictor<Target,
                               Precision>::~PaddleInferenceAnakinPredictor() {
#ifdef PADDLE_ANAKIN_ENABLE_OP_TIMER
  DisplayOpTimer<Target, Precision>(executor_p_, max_batch_size_);
#endif
  delete executor_p_;
  executor_p_ = nullptr;
}

}  // namespace paddle
