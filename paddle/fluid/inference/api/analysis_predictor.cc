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

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/timer.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope) {
  VLOG(3) << "Predictor::init()";
#if !defined(_WIN32)
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";
    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }
#endif

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
    LOG(WARNING) << "ir optimize only supports CPU currently";
    config_.enable_ir_optim = false;
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  if (parent_scope) {
    scope_ = parent_scope;
    sub_scope_ = &(parent_scope->NewScope());
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
  }

  executor_.reset(new paddle::framework::NaiveExecutor(place_));

  if (!LoadProgramDesc()) return false;

  OptimizeInferenceProgram();
  executor_->Prepare(scope_.get(), *inference_program_, 0,
                     config_.use_feed_fetch_ops);

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  VLOG(3) << "Predictor::predict";
  inference::Timer timer;
  timer.tic();
  // set feed variable
  std::vector<framework::LoDTensor> feeds;
  framework::Scope *scope = sub_scope_ != nullptr ? sub_scope_ : scope_.get();
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
  // Run the inference program
  // if share variables, we need not create variables
  executor_->Run();

  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }
  VLOG(3) << "predict cost: " << timer.toc() << "ms";
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }

  LOG(INFO) << "feeds";
  for (auto &item : feed_names_) {
    LOG(INFO) << "name: " << item.first << " " << item.second;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    framework::LoDTensor input;
    framework::DDim ddim = framework::make_ddim(inputs[i].shape);
    void *input_ptr;
    if (inputs[i].dtype == PaddleDType::INT64) {
      input_ptr = input.mutable_data<int64_t>(ddim, platform::CPUPlace());
    } else if (inputs[i].dtype == PaddleDType::FLOAT32) {
      input_ptr = input.mutable_data<float>(ddim, platform::CPUPlace());
    } else {
      LOG(ERROR) << "unsupported feed type " << inputs[i].dtype;
      return false;
    }

    // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
    std::memcpy(static_cast<void *>(input_ptr), inputs[i].data.data(),
                inputs[i].data.length());
    // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
    framework::LoD lod;
    for (auto &level : inputs[i].lod) {
      lod.emplace_back(level);
    }
    input.set_lod(lod);
    int idx = -1;
    if (config_.specify_input_name) {
      idx = feed_names_[inputs[i].name];
    } else {
      idx = boost::get<int>(feeds_[i]->GetAttr("col"));
    }
    LOG(INFO) << "specify " << inputs[i].name << " " << idx;
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const framework::LoDTensor &fetch,
                                    PaddleTensor *output) {
  std::vector<int> shape;
  auto dims_i = fetch.dims();
  auto lod = fetch.lod();
  const T *output_ptr = fetch.data<T>();
  auto num = fetch.numel();
  std::vector<T> data;
  if (0 == lod.size()) {
    std::copy(output_ptr, output_ptr + num, std::back_inserter(data));
    for (int j = 0; j < dims_i.size(); ++j) {
      shape.push_back(dims_i[j]);
    }
  } else {
    // for batch detection
    // image[0] -> output[0] shape {145, 6}
    // image[1] -> output[1] shape {176, 6}
    // then,
    // the batch output shape {321, 6}
    // the lod {{0, 145, 321}}
    // so we should append output[0] to {176, 6}
    size_t max_dim = 0;
    for (size_t j = 1; j < lod[0].size(); j++) {
      max_dim = std::max(max_dim, lod[0][j] - lod[0][j - 1]);
    }
    size_t common_dim = lod[0].back() == 0 ? 0 : num / lod[0].back();
    if (max_dim > 0) {
      data.resize((lod[0].size() - 1) * max_dim * common_dim, 0);
    }
    for (size_t j = 1; j < lod[0].size(); j++) {
      size_t start = lod[0][j - 1] * common_dim;
      size_t end = lod[0][j] * common_dim;
      if (end > start) {
        std::copy(output_ptr + start, output_ptr + end,
                  data.begin() + (j - 1) * max_dim * common_dim);
      }
    }
    shape.push_back(lod[0].size() - 1);
    shape.push_back(max_dim);
    for (int j = 1; j < dims_i.size(); ++j) {
      shape.push_back(dims_i[j]);
    }
  }

  output->shape = shape;
  auto &buffer = output->data;
  if (buffer.empty() || buffer.length() < sizeof(T) * data.size()) {
    buffer.Resize(sizeof(T) * data.size());
  }
  std::memcpy(buffer.data(), data.data(), buffer.length());
  // copy LoD
  for (const auto &level : fetch.lod()) {
    output->lod.emplace_back(level);
  }
}

bool AnalysisPredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetchs_.size());
  for (size_t i = 0; i < fetchs_.size(); ++i) {
    int idx = boost::get<int>(fetchs_[i]->GetAttr("col"));
    PADDLE_ENFORCE((size_t)idx == i);
    framework::LoDTensor &fetch =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    if (type == typeid(float)) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == typeid(int64_t)) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else {
      LOG(ERROR) << "unknown type, only support float32 and int64 now.";
    }
  }
  return true;
}

void AnalysisPredictor::OptimizeInferenceProgram() {
  LOG(INFO) << "optimize begin";
  FLAGS_IA_enable_ir = config_.enable_ir_optim;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "";  // Don't output the model.
  // Analyze inference_program
  if (!config_.model_dir.empty()) {
    argument_.fluid_model_dir.reset(new std::string(config_.model_dir));
  } else {
    PADDLE_ENFORCE(
        !config_.param_file.empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file.empty());
    argument_.fluid_model_program_path.reset(
        new std::string(config_.prog_file));
    argument_.fluid_model_param_path.reset(new std::string(config_.param_file));
  }

  argument_.origin_program_desc.reset(
      new ProgramDesc(*inference_program_->Proto()));
  PADDLE_ENFORCE(config_.ir_mode == AnalysisConfig::IrPassMode::kExclude,
                 "Only kExclude is supported yet.");
  Analyzer().DisableIrPasses(config_.ir_passes).Run(&argument_);

  CHECK(argument_.transformed_program_desc);
  VLOG(5) << "to prepare executor";
  inference_program_.reset(
      new framework::ProgramDesc(*argument_.transformed_program_desc));
  if (argument_.Has(framework::ir::kParamScopeAttr)) {
    // Update scope.
    scope_.reset(
        argument_.Release<framework::Scope>(framework::ir::kParamScopeAttr));
  }
  LOG(INFO) << "== optimize end ==";
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    AnalysisConfig, PaddleEngineKind::kAnalysis>(const AnalysisConfig &config) {
  VLOG(3) << "create AnalysisConfig";
  if (config.use_gpu) {
    // 1. GPU memeroy
    PADDLE_ENFORCE_GT(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  if (!dynamic_cast<AnalysisPredictor *>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
  return predictor;
}

void AnalysisPredictor::PrepareFeedFetch() {
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
    } else if (op->Type() == "fetch") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (fetchs_.size() <= static_cast<size_t>(idx)) {
        fetchs_.resize(idx + 1);
      }
      fetchs_[idx] = op;
    }
  }
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  if (executor_->scope()->FindVar(name) == nullptr) {
    return nullptr;
  }
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  if (executor_->scope()->FindVar(name) != 0) {
    return nullptr;
  }
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  return res;
}

bool AnalysisPredictor::ZeroCopyRun() {
  executor_->Run();

  return true;
}

}  // namespace paddle
