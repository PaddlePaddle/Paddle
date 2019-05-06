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
#include "paddle/fluid/inference/api/paddle_api.h"

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
    const contrib::AnakinConfig &config)
    : config_(config) {
  CHECK(Init());
}
template <>
PaddleInferenceAnakinPredictor<anakin::X86>::PaddleInferenceAnakinPredictor(
    const contrib::AnakinConfig &config)
    : config_(config) {
  omp_set_dynamic(0);
  omp_set_num_threads(1);
  mkl_set_num_threads(1);
  CHECK(Init());
}
template <typename Target>
bool PaddleInferenceAnakinPredictor<Target>::Init() {
  if (!(graph_.load(config_.model_file))) {
    LOG(INFO) << "fail to load graph from " << config_.model_file;
    return false;
  }
  auto inputs = graph_.get_ins();
  for (auto &input_str : inputs) {
    if (config_.init_inputs_shape.find(input_str) ==
        config_.init_inputs_shape.end()) {
      LOG(INFO) << input_str << " is not implemented.";
      return false;
    }
    std::vector<int> shape = config_.init_inputs_shape.find(input_str)->second;
    graph_.Reshape(input_str, shape);
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
  if (config_.re_allocable) {
    return this->RunImpl(inputs, output_data);
  } else {
    // Run inputs data that exceeds batch size in batches.
    // 1. Reassign the batch size.
    if (batch_size == -1) {
      if (!inputs[0].lod.empty()) {
        batch_size = inputs[0].lod[0].size() - 1;
      } else {
        batch_size = inputs[0].shape[0];
      }
    }
    // 2. If the data don't need to be batched, run it directly.
    if (batch_size <= config_.init_batch_size) {
      return this->RunImpl(inputs, output_data);
    }
    // 3. Check the batch size and define temporary variables.
    std::vector<PaddleTensor> cur_inputs;
    std::vector<PaddleTensor> outputs_master;
    std::vector<std::vector<paddle::PaddleTensor>> outputs_vec;
    for (const auto &input : inputs) {
      if (!input.lod.empty()) {
        if (input.lod.size() != 1) {
          return false;
        }
        if (input.lod[0].size() - 1 != batch_size) {
          return false;
        }
      } else {
        LOG(INFO) << "Non-lod mode to be implemented.";
        return false;
      }
      PaddleTensor tensor;
      tensor.name = input.name;
      tensor.dtype = PaddleDType::FLOAT32;
      cur_inputs.push_back(tensor);
    }
    for (auto output : *output_data) {
      PaddleTensor tensor;
      tensor.name = output.name;
      outputs_master.push_back(tensor);
    }
    // 4. Batch execution.
    for (size_t start_batch = 0; start_batch < batch_size;) {
      auto end_batch = start_batch + config_.init_batch_size;
      if (end_batch > batch_size) {
        end_batch = batch_size;
      }
      auto cur_outputs = outputs_master;
      for (size_t i = 0; i < inputs.size(); i++) {
        auto start = inputs[i].lod[0][start_batch];
        auto end = inputs[i].lod[0][end_batch];
        std::vector<size_t> offsets;
        for (size_t j = start_batch; j <= end_batch; j++) {
          offsets.push_back(inputs[i].lod[0][j] -
                            inputs[i].lod[0][start_batch]);
        }
        auto mem_start = static_cast<float *>(inputs[i].data.data()) + start;
        cur_inputs[i].data =
            PaddleBuf(mem_start, (end - start) * sizeof(float));
        cur_inputs[i].lod = std::vector<std::vector<size_t>>({offsets});
        cur_inputs[i].shape =
            std::vector<int>({static_cast<int>(end - start), 1, 1, 1});
      }
      if (!this->RunImpl(cur_inputs, &cur_outputs)) {
        return false;
      }
      outputs_vec.push_back(cur_outputs);
      start_batch = end_batch;
    }
    // 5. Copy the results to contiguous memory.
    // Assume that each batch has the same final outputs size.
    auto count = [](const std::vector<int> &v) {
      int cnt = 1;
      for_each(v.begin(), v.end(), [&cnt](int n) { cnt *= n; });
      return cnt;
    };
    for (size_t i = 0; i < output_data->size(); i++) {
      std::vector<int> shape = outputs_vec[i][0].shape;
      shape[0] = batch_size;
      int total_cnt = count(shape);
      (*output_data)[i].shape = shape;
      (*output_data)[i].data.Resize(total_cnt * sizeof(float));
      float *addr = static_cast<float *>((*output_data)[i].data.data());
      for (const auto &single_out : outputs_vec) {
        int cnt = count(single_out[i].shape);
        memcpy(addr, single_out[i].data.data(), cnt * sizeof(float));
        addr += cnt;
      }
    }
  }
  return true;
}

template <typename Target>
bool PaddleInferenceAnakinPredictor<Target>::RunImpl(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data) {
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
      if (config_.re_allocable) {
        graph_.Reshape(input.name, input.shape);
        delete executor_p_;
        executor_p_ = new anakin::Net<Target, anakin::Precision::FP32,
                                      ::anakin::OpRunType::ASYNC>(graph_, true);
        d_tensor_in_p = executor_p_->get_in(input.name);
      } else {
        LOG(INFO) << "Run failed because Anakin was expected not to reallocate "
                     "memory.";
        return false;
      }
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
      std::vector<int> lod(input.lod[0].begin(), input.lod[0].end());
      std::vector<std::vector<int>> offset({lod});
      d_tensor_in_p->set_seq_offset(offset);
      LOG(INFO) << "offset.size(): " << offset[0].size();
      for (int i = 0; i < offset[0].size(); i++) {
        LOG(INFO) << offset[0][i];
      }
    }

    float *d_data_p = static_cast<float *>(d_tensor_in_p->mutable_data());
#ifdef PADDLE_WITH_CUDA
    if (std::is_same<anakin::NV, Target>::value) {
      if (cudaMemcpy(d_data_p, static_cast<float *>(input.data.data()),
                     d_tensor_in_p->valid_size() * sizeof(float),
                     cudaMemcpyHostToDevice) != 0) {
        LOG(INFO) << "copy data from CPU to GPU error";
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
    LOG(INFO) << "At least one output should be set with tensors' names.";
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
        LOG(INFO) << "copy data from GPU to CPU error";
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
  LOG(INFO) << "Anakin Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(
      new PaddleInferenceAnakinPredictor<Target>(config_));
  // construct executer from other graph
  auto anakin_predictor_p =
      dynamic_cast<PaddleInferenceAnakinPredictor<Target> *>(cls.get());
  if (!anakin_predictor_p) {
    LOG(INFO) << "fail to call Init";
    return nullptr;
  }
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
  LOG(INFO) << "Anakin Predictor create.";
  if (config.target_type == contrib::AnakinConfig::NVGPU) {
#ifdef PADDLE_WITH_CUDA
    LOG(INFO) << "Anakin Predictor create on [ NVIDIA GPU ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::NV>(config));
    return x;
#else
    LOG(ERROR) << "AnakinConfig::NVGPU could not used in ONLY-CPU environment";
    return nullptr;
#endif
  } else if (config.target_type == contrib::AnakinConfig::X86) {
    LOG(INFO) << "Anakin Predictor create on [ Intel X86 ].";
    std::unique_ptr<PaddlePredictor> x(
        new PaddleInferenceAnakinPredictor<anakin::X86>(config));
    return x;
  } else {
    LOG(INFO) << "Anakin Predictor create on unknown platform.";
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
  DisplayOpTimer<Target>(executor_p_, config_.init_batch_size);
#endif
  delete executor_p_;
  executor_p_ = nullptr;
}

}  // namespace paddle
