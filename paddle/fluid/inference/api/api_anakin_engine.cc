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

#include "paddle/fluid/inference/api/api_anakin_engine.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#include <mkl_service.h>
#include <omp.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "framework/core/net/net.h"
#include "framework/operators/ops.h"
#include "saber/funcs/timer.h"

namespace paddle {

using paddle::contrib::AnakinConfig;

template <typename Target>
PaddleInferenceAnakinPredictor<Target>::PaddleInferenceAnakinPredictor(
    const contrib::AnakinConfig &config) {
  CHECK(Init(config));
}
template <>
PaddleInferenceAnakinPredictor<anakin::X86>::PaddleInferenceAnakinPredictor(
    const contrib::AnakinConfig &config) {
  omp_set_dynamic(0);
  omp_set_num_threads(1);
  mkl_set_num_threads(1);
  CHECK(Init(config));
}
template <typename Target>
bool PaddleInferenceAnakinPredictor<Target>::Init(
    const contrib::AnakinConfig &config) {
  if (!(graph_.load(config.model_file))) {
    VLOG(3) << "fail to load graph from " << config.model_file;
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
    executor_p_ = new anakin::Net<Target, anakin::Precision::FP32,
                                  ::anakin::OpRunType::ASYNC>(graph_, true);
  }
  return true;
}

template <typename Target>
bool PaddleInferenceAnakinPredictor<Target>::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data, int batch_size) {
  for (const auto &input : inputs) {
    if (input.dtype != PaddleDType::FLOAT32) {
      VLOG(3) << "Only support float type inputs. " << input.name
              << "'s type is not float";
      return false;
    }
    auto d_tensor_in_p = executor_p_->get_in(input.name);
    auto net_shape = d_tensor_in_p->shape();
    if (net_shape.size() != input.shape.size()) {
      VLOG(3) << " input  " << input.name
              << "'s shape size should be equal to that of net";
      return false;
    }
    int sum = 1;
    for_each(input.shape.begin(), input.shape.end(), [&](int n) { sum *= n; });
    if (sum > net_shape.count()) {
      graph_.Reshape(input.name, input.shape);
      delete executor_p_;
      executor_p_ = new anakin::Net<Target, anakin::Precision::FP32,
                                    ::anakin::OpRunType::ASYNC>(graph_, true);
      d_tensor_in_p = executor_p_->get_in(input.name);
    }

    anakin::saber::Shape tmp_shape;
    for (auto s : input.shape) {
      tmp_shape.push_back(s);
    }
    d_tensor_in_p->reshape(tmp_shape);

    if (input.lod.size() > 0) {
      if (input.lod.size() > 1) {
        VLOG(3) << " input lod first dim should <=1, but you set "
                << input.lod.size();
        return false;
      }
      std::vector<int> lod(input.lod[0].begin(), input.lod[0].end());
      std::vector<std::vector<int>> offset({lod});
      d_tensor_in_p->set_seq_offset(offset);
      VLOG(3) << "offset.size(): " << offset[0].size();
      for (int i = 0; i < offset[0].size(); i++) {
        VLOG(3) << offset[0][i];
      }
    }

    float *d_data_p = static_cast<float *>(d_tensor_in_p->mutable_data());

#ifdef PADDLE_WITH_CUDA
    if (std::is_same<anakin::NV, Target>::value) {
      if (cudaMemcpy(d_data_p, static_cast<float *>(input.data.data()),
                     d_tensor_in_p->valid_size() * sizeof(float),
                     cudaMemcpyHostToDevice) != 0) {
        VLOG(3) << "copy data from CPU to GPU error";
        return false;
      }
    }
#endif
    if (std::is_same<anakin::X86, Target>::value) {
      memcpy(d_data_p, static_cast<float *>(input.data.data()),
             d_tensor_in_p->valid_size() * sizeof(float));
    }
  }
#ifdef PADDLE_WITH_CUDA
  cudaDeviceSynchronize();
  executor_p_->prediction();
  cudaDeviceSynchronize();
#endif

  if (output_data->empty()) {
    VLOG(3) << "At least one output should be set with tensors' names.";
    return false;
  }
  for (auto &output : *output_data) {
    auto *tensor = executor_p_->get_out(output.name);
    output.shape = tensor->valid_shape();
    if (output.data.length() < tensor->valid_size() * sizeof(float)) {
      output.data.Resize(tensor->valid_size() * sizeof(float));
    }

#if PADDLE_WITH_CUDA
    if (std::is_same<anakin::NV, Target>::value) {
      // Copy data from GPU -> CPU
      if (cudaMemcpy(output.data.data(), tensor->mutable_data(),
                     tensor->valid_size() * sizeof(float),
                     cudaMemcpyDeviceToHost) != 0) {
        VLOG(3) << "copy data from GPU to CPU error";
        return false;
      }
    }
#endif
    if (std::is_same<anakin::X86, Target>::value) {
      memcpy(output.data.data(), tensor->mutable_data(),
             tensor->valid_size() * sizeof(float));
    }
  }
  return true;
}

template <typename Target>
anakin::Net<Target, anakin::Precision::FP32, ::anakin::OpRunType::ASYNC>
    &PaddleInferenceAnakinPredictor<Target>::get_executer() {
  return *executor_p_;
}

// the cloned new Predictor of anakin share the same net weights from original
// Predictor
template <typename Target>
std::unique_ptr<PaddlePredictor>
PaddleInferenceAnakinPredictor<Target>::Clone() {
  VLOG(3) << "Anakin Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(
      new PaddleInferenceAnakinPredictor<Target>());
  // construct executer from other graph
  auto anakin_predictor_p =
      dynamic_cast<PaddleInferenceAnakinPredictor<Target> *>(cls.get());
  if (!anakin_predictor_p) {
    VLOG(3) << "fail to call Init";
    return nullptr;
  }
  anakin_predictor_p->get_executer().init(graph_);

  return std::move(cls);
}

#ifdef PADDLE_WITH_CUDA
template class PaddleInferenceAnakinPredictor<anakin::NV>;
#endif
template class PaddleInferenceAnakinPredictor<anakin::X86>;

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
    const contrib::AnakinConfig &config) {
  VLOG(3) << "Anakin Predictor create.";
  if (config.target_type == contrib::AnakinConfig::NVGPU) {
#ifdef PADDLE_WITH_CUDA
    VLOG(3) << "Anakin Predictor create on [ NVIDIA GPU ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::NV>(config));
    return x;
#else
    LOG(ERROR) << "AnakinConfig::NVGPU could not used in ONLY-CPU environment";
    return nullptr;
#endif
  } else if (config.target_type == contrib::AnakinConfig::X86) {
    VLOG(3) << "Anakin Predictor create on [ Intel X86 ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::X86>(config));
    return x;
  } else {
    VLOG(3) << "Anakin Predictor create on unknown platform.";
    return nullptr;
  }
}

#ifdef PADDLE_ANAKIN_ENABLE_OP_TIMER
template <typename Target>
using executor_t =
    anakin::Net<Target, anakin::saber::AK_FLOAT, anakin::Precision::FP32>;

template <typename Target>
void DisplayOpTimer(executor_t<Target> *net_executor, int epoch) {
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

template <typename Target>
PaddleInferenceAnakinPredictor<Target>::~PaddleInferenceAnakinPredictor() {
#ifdef PADDLE_ANAKIN_ENABLE_OP_TIMER
  DisplayOpTimer<Target>(executor_p_, max_batch_size_);
#endif
  delete executor_p_;
  executor_p_ = nullptr;
}

}  // namespace paddle
