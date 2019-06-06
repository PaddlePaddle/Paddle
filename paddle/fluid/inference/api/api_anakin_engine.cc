// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/inference/api/api_anakin_engine.h"
#include "paddle/fluid/inference/api/paddle_api.h"

#include "framework/core/net/net.h"
#include "framework/operators/ops.h"
#include "saber/funcs/timer.h"

namespace paddle {

using paddle::contrib::AnakinConfig;
template <typename T, Precision P, OpRunType R>
extern std::mutex PaddleInferenceAnakinPredictor<T, P, R>::mutex_;
template <typename T, Precision P, OpRunType R>
extern std::once_flag PaddleInferenceAnakinPredictor<T, P, R>::init_anakin_;

template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::InitEnv() {
  anakin::TargetWrapper<T>::set_device(this->config_.device_id);
  std::call_once(this->init_anakin_, [this]() {
    anakin::Env<T>::env_init(this->config_.max_stream);
  });
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::InitNet() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  this->executor_p_ = new anakin::Net<T, P, R>(*this->graph_p_, true);
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::SetContext() {
  this->ctx_p_ = std::make_shared<anakin::Context<T>>(
      this->config_.device_id, this->config_.data_stream_id,
      this->config_.compute_stream_id);
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::InitGraph() {
  this->graph_p_ =
      std::make_shared<anakin::graph::Graph<T, anakin::Precision::FP32>>();
  if (!(this->graph_p_->load(this->config_.model_file))) {
    LOG(FATAL) << "fail to load graph from " << this->config_.model_file;
  }
  auto inputs = this->graph_p_->get_ins();
  for (auto &input_str : inputs) {
    if (this->config_.init_inputs_shape.find(input_str) ==
        this->config_.init_inputs_shape.end()) {
      LOG(FATAL) << input_str << " is not implemented.";
    }
    std::vector<int> shape =
        this->config_.init_inputs_shape.find(input_str)->second;
    this->graph_p_->Reshape(input_str, shape);
  }
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::OptimizeGraph() {
  if (!this->graph_p_->Optimize()) {
    LOG(FATAL) << "Graph optimization error.";
  }
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::InitPredictor() {
  this->InitEnv();
  this->SetContext();
  this->InitGraph();
  this->OptimizeGraph();
  this->InitNet();
}
template <typename T, Precision P, OpRunType R>
void PaddleInferenceAnakinPredictor<T, P, R>::Predict() {
  anakin::TargetWrapper<T>::device_sync();
  this->executor_p_->prediction();
  anakin::TargetWrapper<T>::device_sync();
}
template <typename T, Precision P, OpRunType R>
bool PaddleInferenceAnakinPredictor<T, P, R>::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data, int batch_size) {
  if (this->config_.re_allocable) {
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
    if (batch_size <= this->config_.init_batch_size) {
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
      auto end_batch = start_batch + this->config_.init_batch_size;
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
template <typename T, Precision P, OpRunType R>
bool PaddleInferenceAnakinPredictor<T, P, R>::RunImpl(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data) {
  for (const auto &input : inputs) {
    if (input.dtype != PaddleDType::FLOAT32) {
      LOG(FATAL) << "Only support float type inputs. " << input.name
                 << "'s type is not float";
    }
    auto d_tensor_p = this->executor_p_->get_in(input.name);
    auto net_shape = d_tensor_p->shape();
    if (net_shape.size() != input.shape.size()) {
      LOG(FATAL) << " input  " << input.name
                 << "'s shape size should be equal to that of net";
    }
    int sum = 1;
    for_each(input.shape.begin(), input.shape.end(), [&](int n) { sum *= n; });
    if (sum > net_shape.count()) {
      if (this->config_.re_allocable) {
        this->graph_p_->Reshape(input.name, input.shape);
        delete this->executor_p_;
        this->InitNet();
        d_tensor_p = this->executor_p_->get_in(input.name);
      } else {
        LOG(FATAL)
            << "Run failed because Anakin was expected not to reallocate "
               "memory.";
      }
    }
    std::vector<int> tmp_shape;
    for (auto s : input.shape) {
      tmp_shape.push_back(s);
    }
    auto *data = static_cast<float *>(input.data.data());
    anakin::saber::Tensor<typename anakin::DefaultHostType<T>::Host_type>
        h_tensor(data, typename anakin::DefaultHostType<T>::Host_type(), 0,
                 tmp_shape);
    d_tensor_p->reshape(tmp_shape);

    if (input.lod.size() > 0) {
      if (input.lod.size() > 1) {
        LOG(FATAL) << " input lod first dim should <=1, but you set "
                   << input.lod.size();
      }
      std::vector<int> lod(input.lod[0].begin(), input.lod[0].end());
      std::vector<std::vector<int>> offset({lod});
      d_tensor_p->set_seq_offset(offset);
      VLOG(3) << "offset.size(): " << offset[0].size();
      for (int i = 0; i < offset[0].size(); i++) {
        VLOG(3) << offset[0][i];
      }
    }
    d_tensor_p->copy_from(h_tensor);
  }
  this->Predict();
  if (output_data->empty()) {
    LOG(FATAL) << "At least one output should be set with tensors' names.";
  }
  for (auto &output : *output_data) {
    auto *d_tensor_p = this->executor_p_->get_out(output.name);
    output.shape = d_tensor_p->valid_shape();
    if (output.data.length() < d_tensor_p->valid_size() * sizeof(float)) {
      output.data.Resize(d_tensor_p->valid_size() * sizeof(float));
    }
    auto *data = static_cast<float *>(output.data.data());
    anakin::saber::Tensor<typename anakin::DefaultHostType<T>::Host_type>
        h_tensor(data, typename anakin::DefaultHostType<T>::Host_type(), 0,
                 d_tensor_p->valid_shape());
    h_tensor.copy_from(*d_tensor_p);
  }
  return true;
}
template <typename T, Precision P, OpRunType R>
bool PaddleInferenceAnakinPredictor<T, P, R>::ResetConfig(
    const AnakinConfig &config) {
  this->config_ = config;
  return true;
}
template <typename T, Precision P, OpRunType R>
anakin::Net<T, P, R> &PaddleInferenceAnakinPredictor<T, P, R>::ResetExecuter(
    std::shared_ptr<anakin::graph::Graph<T, P>> graph_p) {
  this->graph_p_ = graph_p;
  this->ctx_p_ = std::make_shared<anakin::Context<T>>(
      this->config_.device_id, this->config_.data_stream_id,
      this->config_.compute_stream_id);
  this->InitNet();
  return *this->executor_p_;
}
// the cloned new Predictor of anakin share the same net weights from original
// Predictor
template <typename T, Precision P, OpRunType R>
std::unique_ptr<PaddlePredictor>
PaddleInferenceAnakinPredictor<T, P, R>::Clone() {
  VLOG(3) << "Anakin Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(
      new PaddleInferenceAnakinPredictor<T, P, R>());
  // construct executer from other graph
  auto anakin_predictor_p =
      dynamic_cast<PaddleInferenceAnakinPredictor<T, P, R> *>(cls.get());
  if (!anakin_predictor_p) {
    LOG(FATAL) << "fail to call Init";
  }
  anakin_predictor_p->ResetConfig(this->config_);
  anakin_predictor_p->ResetExecuter(this->graph_p_);
  return cls;
}

#ifdef ANAKIN_MLU_PLACE
template <Precision P, OpRunType R>
void PaddleInferenceAnakinMLUPredictor<P, R>::SetContext() {
  this->ctx_p_ = std::make_shared<anakin::Context<anakin::MLU>>(
      this->config_.device_id, this->config_.data_stream_id,
      this->config_.compute_stream_id);
  this->ctx_p_->set_model_parallel(this->config_.model_parallel);
  this->ctx_p_->set_fusion(this->config_.op_fuse);
}
template <Precision P, OpRunType R>
void PaddleInferenceAnakinMLUPredictor<P, R>::OptimizeGraph() {
  if (!this->graph_p_->fusion_optimize(this->config_.op_fuse)) {
    LOG(FATAL) << "Graph optimization error.";
  }
}
template <Precision P, OpRunType R>
void PaddleInferenceAnakinMLUPredictor<P, R>::InitNet() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  this->executor_p_ = new anakin::Net<anakin::MLU, P, R>();
  this->executor_p_->fusion_init(*this->graph_p_, this->ctx_p_, true);
}
template <Precision P, OpRunType R>
void PaddleInferenceAnakinMLUPredictor<P, R>::Predict() {
  anakin::TargetWrapper<anakin::MLU>::device_sync();
  this->executor_p_->fusion_prediction();
  anakin::TargetWrapper<anakin::MLU>::device_sync();
}
#endif

#ifdef PADDLE_WITH_CUDA
template class PaddleInferenceAnakinPredictor<
    anakin::NV, anakin::Precision::FP32, ::anakin::OpRunType::ASYNC>;
#endif
#ifdef ANAKIN_X86_PLACE
template class PaddleInferenceAnakinPredictor<
    anakin::X86, anakin::Precision::FP32, ::anakin::OpRunType::ASYNC>;
#endif
#ifdef ANAKIN_MLU_PLACE
template class PaddleInferenceAnakinMLUPredictor<anakin::Precision::FP32,
                                                 ::anakin::OpRunType::SYNC>;
#endif

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
    const contrib::AnakinConfig &config) {
#ifdef PADDLE_WITH_CUDA
  if (config.target_type == contrib::AnakinConfig::NVGPU) {
    return std::unique_ptr<PaddlePredictor>(
        new PaddleInferenceAnakinPredictor<anakin::NV, anakin::Precision::FP32,
                                           ::anakin::OpRunType::ASYNC>(config));
  }
#endif
#ifdef ANAKIN_X86_PLACE
  if (config.target_type == contrib::AnakinConfig::X86) {
    return std::unique_ptr<PaddlePredictor>(
        new PaddleInferenceAnakinPredictor<anakin::X86, anakin::Precision::FP32,
                                           ::anakin::OpRunType::ASYNC>(config));
  }
#endif
#ifdef ANAKIN_MLU_PLACE
  if (config.target_type == contrib::AnakinConfig::MLU) {
    return std::unique_ptr<PaddlePredictor>(
        new PaddleInferenceAnakinMLUPredictor<anakin::Precision::FP32,
                                              ::anakin::OpRunType::SYNC>(
            config));
  }
#endif
  LOG(FATAL) << "Anakin Predictor create on unknown platform.";
  return nullptr;
}
template <typename T, Precision P, OpRunType R>
void DisplayOpTimer(anakin::Net<T, P, R> *net_executor, int epoch) {
#ifdef PADDLE_ANAKIN_ENABLE_OP_TIMER
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
#endif
}
template <typename T, Precision P, OpRunType R>
PaddleInferenceAnakinPredictor<T, P, R>::~PaddleInferenceAnakinPredictor() {
  DisplayOpTimer<T, P, R>(this->executor_p_, this->config_.init_batch_size);
  delete this->executor_p_;
  this->executor_p_ = nullptr;
}

}  // namespace paddle
