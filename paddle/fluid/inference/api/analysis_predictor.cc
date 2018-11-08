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
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {

using contrib::AnalysisConfig;

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
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

  if (!program) {
    if (!LoadProgramDesc()) return false;
    OptimizeInferenceProgram();
  } else {
    inference_program_ = program;
  }

  if (config_._use_mkldnn) {
    executor_->EnableMKLDNN(*inference_program_);
  }

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
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
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

  // Cache the inputs memory for better concurrency performance.
  feed_tensors_.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = feed_tensors_[i];
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
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const framework::LoDTensor &fetch,
                                    PaddleTensor *output) {
  // set shape.
  auto shape = framework::vectorize(fetch.dims());
  output->shape.assign(shape.begin(), shape.end());
  // set data.
  const T *data = fetch.data<T>();
  int num_elems = inference::VecReduceToInt(shape);
  output->data.Resize(num_elems * sizeof(T));
  // The fetched tensor output by fetch op, should always in CPU memory, so just
  // copy.
  memcpy(output->data.data(), data, num_elems * sizeof(T));
  // set lod
  output->lod.clear();
  for (auto &level : fetch.lod()) {
    output->lod.emplace_back(level.begin(), level.end());
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

// NOTE All the members in AnalysisConfig should be copied to Argument.
void AnalysisPredictor::OptimizeInferenceProgram() {
  argument_.SetUseGPU(config_.use_gpu);
  // Analyze inference_program
  if (!config_.model_dir.empty()) {
    argument_.SetModelDir(config_.model_dir);
  } else {
    PADDLE_ENFORCE(
        !config_.param_file.empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file.empty());
    argument_.SetModelProgramPath(config_.prog_file);
    argument_.SetModelParamsPath(config_.param_file);
  }

  if (config_.use_gpu && config_.use_tensorrt_) {
    LOG(INFO) << "argument_ set TensorRT true";
    argument_.SetUseTensorRT(true);
    argument_.SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_.SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
  }

  auto passes = config_.pass_builder()->AllPasses();
  if (!config_.enable_ir_optim) passes.clear();
  argument_.SetIrAnalysisPasses(passes);
  argument_.SetScopeNotOwned(const_cast<framework::Scope *>(scope_.get()));
  Analyzer().Run(&argument_);

  PADDLE_ENFORCE(argument_.scope_valid());
  LOG(INFO) << "analyzed scope has vars " << argument_.scope().LocalVarNames().size();

  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((&argument_), ir_analyzed_program);
  inference_program_.reset(
      new framework::ProgramDesc(argument_.ir_analyzed_program()));
  if (argument_.scope_valid()) {
    // Update scope.
    scope_.reset(argument_.ReleaseScope());
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
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
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

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::unique_ptr<framework::Executor> tmp_exe(
      new framework::Executor(platform::CPUPlace()));
  if (!config_.model_dir.empty()) {
    // Parameters are saved in separate files sited in
    // the specified `dirname`.
    inference_program_ = paddle::inference::Load(
        static_cast<framework::Executor *>(tmp_exe.get()), scope_.get(),
        config_.model_dir);
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    inference_program_ = paddle::inference::Load(
        static_cast<framework::Executor *>(tmp_exe.get()), scope_.get(),
        config_.prog_file, config_.param_file);
  } else {
    if (config_.model_dir.empty() && config_.prog_file.empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.", config_.model_dir,
        config_.param_file);
    return false;
  }
  return true;
}
std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone() {
  auto *x = new AnalysisPredictor(config_);
  x->Init(scope_, inference_program_);
  return std::unique_ptr<PaddlePredictor>(x);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<contrib::AnalysisConfig>(
    const contrib::AnalysisConfig &config) {
  return CreatePaddlePredictor<contrib::AnalysisConfig,
                               PaddleEngineKind::kAnalysis>(config);
}

}  // namespace paddle

// TODO(Superjomn) add an definition PADDLE_WITH_TENSORRT.
#if PADDLE_WITH_CUDA
USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(elementwise_add_tensor);
USE_TRT_CONVERTER(elementwise_sub_tensor);
USE_TRT_CONVERTER(elementwise_div_tensor);
USE_TRT_CONVERTER(elementwise_mul_tensor);
USE_TRT_CONVERTER(elementwise_max_tensor);
USE_TRT_CONVERTER(elementwise_min_tensor);
USE_TRT_CONVERTER(elementwise_pow_tensor);
USE_TRT_CONVERTER(mul);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(sigmoid);
USE_TRT_CONVERTER(tanh);
USE_TRT_CONVERTER(fc);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
USE_TRT_CONVERTER(batch_norm);
USE_TRT_CONVERTER(concat);
USE_TRT_CONVERTER(dropout);
USE_TRT_CONVERTER(pad);
#endif
