/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(profile, false, "Turn on profiler for fluid");

namespace paddle {
namespace {
using paddle::inference::Timer;

template <class T>
std::string num2str(T a) {
  std::stringstream istr;
  istr << a;
  return istr.str();
}
}  // namespace

void NativePaddlePredictor::PrepareFeedFetch() {
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
    } else if (op->Type() == "fetch") {
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (fetchs_.size() <= static_cast<size_t>(idx)) {
        fetchs_.resize(idx + 1);
      }
      fetchs_[idx] = op;
    }
  }
}

bool NativePaddlePredictor::Init(
    std::shared_ptr<framework::Scope> parent_scope) {
  VLOG(3) << "Predictor::init()";
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";

    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  if (parent_scope) {
    scope_ = parent_scope;
    sub_scope_ = &(parent_scope->NewScope());
    PADDLE_ENFORCE_NOT_NULL(sub_scope_, "create sub scope fail");
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
  }

  executor_.reset(new paddle::framework::Executor(place_));

  // Initialize the inference program
  if (!config_.model_dir.empty()) {
    // Parameters are saved in separate files sited in
    // the specified `dirname`.
    inference_program_ = paddle::inference::Load(executor_.get(), scope_.get(),
                                                 config_.model_dir);
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    inference_program_ = paddle::inference::Load(
        executor_.get(), scope_.get(), config_.prog_file, config_.param_file);
  } else {
    LOG(ERROR) << "fail to load inference model from " << config_.model_dir;
    return false;
  }

  ctx_ = executor_->Prepare(*inference_program_, 0);
  executor_->CreateVariables(*inference_program_,
                             sub_scope_ ? sub_scope_ : scope_.get(), 0);

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

NativePaddlePredictor::~NativePaddlePredictor() {
  if (FLAGS_profile) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }
}

bool NativePaddlePredictor::Run(const std::vector<PaddleTensor> &inputs,
                                std::vector<PaddleTensor> *output_data,
                                int batch_size) {
#ifndef PADDLE_ON_INFERENCE
  LOG_FIRST_N(WARNING, 5) << "The NaiveExecutor can not work properly if the "
                             "cmake flag ON_INFER is not set.";
  LOG_FIRST_N(WARNING, 5) << "Unlike the training phase, all the scopes and "
                             "variables will be reused to save the allocation "
                             "overhead.";
  LOG_FIRST_N(WARNING, 5) << "Please re-compile the inference library by "
                             "setting the cmake flag ON_INFER=ON if you are "
                             "running Paddle Inference";
#endif  // PADDLE_ON_INFERENCE
  if (UNLIKELY(config_.cpu_math_library_num_threads() > 1)) {
    paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
  }
  VLOG(3) << "Predictor::predict";
  Timer timer;
  timer.tic();
  // set feed variable
  framework::Scope *scope = sub_scope_ != nullptr ? sub_scope_ : scope_.get();
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
  // Run the inference program
  // if share variables, we need not create variables
  VLOG(4) << "Run prepared context";
  executor_->RunPreparedContext(ctx_.get(), scope,
                                false, /* don't create local scope each time*/
                                false /* don't create variable each time */);
  VLOG(4) << "Finish prepared context";
  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }
  VLOG(3) << "predict cost: " << timer.toc() << "ms";

  // For some other vector like containers not cleaned after each batch.
  tensor_array_batch_cleaner_.CollectNoTensorVars(scope_.get());
  tensor_array_batch_cleaner_.ResetNoTensorVars();
  return true;
}

std::unique_ptr<PaddlePredictor> NativePaddlePredictor::Clone() {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  VLOG(3) << "Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(new NativePaddlePredictor(config_));
  // Hot fix the bug that result diff in multi-thread.
  // TODO(Superjomn) re-implement a real clone here.
  PADDLE_ENFORCE_NOT_NULL(dynamic_cast<NativePaddlePredictor *>(cls.get()));
  if (!dynamic_cast<NativePaddlePredictor *>(cls.get())->Init(nullptr)) {
    LOG(ERROR) << "fail to call Init";
    return nullptr;
  }

#ifdef __clang__
  // fix clang compile error
  return cls;
#else
  // fix manylinux compile error.
  return std::move(cls);
#endif
}

bool NativePaddlePredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
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
      input_ptr = input.mutable_data<int64_t>(ddim, place_);
    } else if (inputs[i].dtype == PaddleDType::FLOAT32) {
      input_ptr = input.mutable_data<float>(ddim, place_);
    } else if (inputs[i].dtype == PaddleDType::INT32) {
      input_ptr = input.mutable_data<int32_t>(ddim, place_);
    } else {
      LOG(ERROR) << "unsupported feed type " << inputs[i].dtype;
      return false;
    }

    PADDLE_ENFORCE_NOT_NULL(input_ptr);
    PADDLE_ENFORCE_NOT_NULL(inputs[i].data.data());
    if (platform::is_cpu_place(place_)) {
      // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
      std::memcpy(static_cast<void *>(input_ptr), inputs[i].data.data(),
                  inputs[i].data.length());
    } else {
#ifdef PADDLE_WITH_CUDA
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto *dev_ctx =
          static_cast<const platform::CUDADeviceContext *>(pool.Get(place_));
      auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place_);
      memory::Copy(dst_gpu_place, static_cast<void *>(input_ptr),
                   platform::CPUPlace(), inputs[i].data.data(),
                   inputs[i].data.length(), dev_ctx->stream());
#else
      PADDLE_THROW("Not compile with CUDA, should not reach here.");
#endif
    }

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
      idx = BOOST_GET_CONST(int, feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}
template <typename T>
void NativePaddlePredictor::GetFetchOne(const framework::LoDTensor &fetch,
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

bool NativePaddlePredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                     framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetchs_.size());
  for (size_t i = 0; i < fetchs_.size(); ++i) {
    int idx = BOOST_GET_CONST(int, fetchs_[i]->GetAttr("col"));
    PADDLE_ENFORCE((size_t)idx == i);
    framework::FetchType &fetch_var =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto fetch = BOOST_GET_CONST(framework::LoDTensor, fetch_var);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    output->name = fetchs_[idx]->Input("X")[0];
    if (type == framework::DataTypeTrait<float>::DataType()) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == framework::DataTypeTrait<int64_t>::DataType()) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else if (type == framework::DataTypeTrait<int32_t>::DataType()) {
      GetFetchOne<int32_t>(fetch, output);
      output->dtype = PaddleDType::INT32;
    } else {
      LOG(ERROR) << "unknown type, only support float32, int64 and int32 now.";
    }
  }
  return true;
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    NativeConfig, PaddleEngineKind::kNative>(const NativeConfig &config) {
  VLOG(3) << "create NativePaddlePredictor";
  if (config.use_gpu) {
    // 1. GPU memory
    PADDLE_ENFORCE_GE(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         num2str<float>(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new NativePaddlePredictor(config));
  PADDLE_ENFORCE_NOT_NULL(
      dynamic_cast<NativePaddlePredictor *>(predictor.get()));
  if (!dynamic_cast<NativePaddlePredictor *>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
#ifdef __clang__
  // fix clang compile error
  return predictor;
#else
  return std::move(predictor);
#endif
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<NativeConfig>(
    const NativeConfig &config) {
  return CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
}

}  // namespace paddle

namespace paddle_infer {

Config::Config(const std::string &model_dir) {
  Init();
  config_.SetModel(model_dir);
}

Config::Config(const std::string &prog_file, const std::string &params_file) {
  Init();
  config_.SetModel(prog_file, params_file);
}

void Config::SetModel(const std::string &model_dir) {
  Init();
  config_.SetModel(model_dir);
}

void Config::SetModel(const std::string &prog_file,
                      const std::string &params_file) {
  Init();
  config_.SetModel(prog_file, params_file);
}

void Config::SetModelBuffer(const char *prog_buffer, size_t prog_buffer_size,
                            const char *params_buffer,
                            size_t params_buffer_size) {
  Init();
  config_.SetModelBuffer(prog_buffer, prog_buffer_size, params_buffer,
                         params_buffer_size);
}

void Config::EnableMemoryOptim() { config_.EnableMemoryOptim(); }

void Config::EnableMKLDNN() { config_.EnableMKLDNN(); }

void Config::SetMkldnnCacheCapacity(int capacity) {
  config_.SetMkldnnCacheCapacity(capacity);
}

void Config::SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads) {
  config_.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
}

void Config::EnableMkldnnQuantizer() { config_.EnableMkldnnQuantizer(); }

paddle::MkldnnQuantizerConfig *Config::mkldnn_quantizer_config() const {
  return config_.mkldnn_quantizer_config();
}

void Config::EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id) {
  config_.EnableUseGpu(memory_pool_init_size_mb, device_id);
}

void Config::EnableTensorRtEngine(int workspace_size, int max_batch_size,
                                  int min_subgraph_size, Precision precision,
                                  bool use_static, bool use_calib_mode) {
  config_.EnableTensorRtEngine(workspace_size, max_batch_size,
                               min_subgraph_size, PrecisionTrait(precision),
                               use_static, use_calib_mode);
}

bool Config::tensorrt_engine_enabled() const {
  return config_.tensorrt_engine_enabled();
}

void Config::SetTRTDynamicShapeInfo(
    std::map<std::string, std::vector<int>> min_input_shape,
    std::map<std::string, std::vector<int>> max_input_shape,
    std::map<std::string, std::vector<int>> optim_input_shape,
    bool disable_trt_plugin_fp16) {
  config_.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                 optim_input_shape, disable_trt_plugin_fp16);
}

void Config::EnableGpuMultiStream() { config_.EnableGpuMultiStream(); }

void Config::EnableLiteEngine(Precision precision, bool zero_copy,
                              const std::vector<std::string> &passes_filter,
                              const std::vector<std::string> &ops_filter) {
  config_.EnableLiteEngine(PrecisionTrait(precision), zero_copy, passes_filter,
                           ops_filter);
}

void Config::EnableXpu(int l3_workspace_size) {
  config_.EnableXpu(l3_workspace_size);
}

void Config::SwitchIrOptim(int x) { config_.SwitchIrOptim(x); }

void Config::EnableProfile() { config_.EnableProfile(); }

void Config::DisableGlogInfo() { config_.DisableGlogInfo(); }

paddle::PassStrategy *Config::pass_builder() const {
  return config_.pass_builder();
}

void Tensor::Reshape(const std::vector<int> &shape) { tensor_->Reshape(shape); }

template <typename T>
void Tensor::CopyFromCpu(const T *data) {
  tensor_->copy_from_cpu<T>(data);
}
template PD_INFER_DECL void Tensor::CopyFromCpu<float>(const float *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<int64_t>(const int64_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<int32_t>(const int32_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<uint8_t>(const uint8_t *data);

template <typename T>
T *Tensor::mutable_data(PaddlePlace place) {
  return tensor_->mutable_data<T>(static_cast<paddle::PaddlePlace>(place));
}

template PD_INFER_DECL float *Tensor::mutable_data<float>(PaddlePlace);
template PD_INFER_DECL int64_t *Tensor::mutable_data<int64_t>(PaddlePlace);
template PD_INFER_DECL int32_t *Tensor::mutable_data<int32_t>(PaddlePlace);
template PD_INFER_DECL uint8_t *Tensor::mutable_data<uint8_t>(PaddlePlace);

template <typename T>
void Tensor::CopyToCpu(T *data) {
  return tensor_->copy_to_cpu<T>(data);
}

template PD_INFER_DECL void Tensor::CopyToCpu<float>(float *data);
template PD_INFER_DECL void Tensor::CopyToCpu<int64_t>(int64_t *data);
template PD_INFER_DECL void Tensor::CopyToCpu<int32_t>(int32_t *data);
template PD_INFER_DECL void Tensor::CopyToCpu<uint8_t>(uint8_t *data);

template <typename T>
T *Tensor::data(PaddlePlace *place, int *size) const {
  paddle::PaddlePlace tmp;
  auto *data = tensor_->data<T>(&tmp, size);
  *place = static_cast<PaddlePlace>(tmp);
  return data;
}

template PD_INFER_DECL float *Tensor::data<float>(PaddlePlace *place,
                                                  int *size) const;
template PD_INFER_DECL int64_t *Tensor::data<int64_t>(PaddlePlace *place,
                                                      int *size) const;
template PD_INFER_DECL int32_t *Tensor::data<int32_t>(PaddlePlace *place,
                                                      int *size) const;
template PD_INFER_DECL uint8_t *Tensor::data<uint8_t>(PaddlePlace *place,
                                                      int *size) const;

std::vector<int> Tensor::shape() const { return tensor_->shape(); }

void Tensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
  return tensor_->SetLoD(x);
}

std::vector<std::vector<size_t>> Tensor::lod() const { return tensor_->lod(); }

const std::string &Tensor::name() const { return tensor_->name(); }

PaddleDType Tensor::type() const {
  return static_cast<PaddleDType>(tensor_->type());
}

std::vector<std::string> Predictor::GetInputNames() {
  return predictor_->GetInputNames();
}

std::unique_ptr<Tensor> Predictor::GetInputHandle(const std::string &name) {
  auto zero_copy_tensor = predictor_->GetInputTensor(name);
  std::unique_ptr<Tensor> tensor(new Tensor(std::move(zero_copy_tensor)));
  return tensor;
}

std::vector<std::string> Predictor::GetOutputNames() {
  return predictor_->GetOutputNames();
}

std::unique_ptr<Tensor> Predictor::GetOutputHandle(const std::string &name) {
  auto zero_copy_tensor = predictor_->GetOutputTensor(name);
  std::unique_ptr<Tensor> tensor(new Tensor(std::move(zero_copy_tensor)));
  return tensor;
}

bool Predictor::Run() { return predictor_->ZeroCopyRun(); }

std::unique_ptr<Predictor> Predictor::Clone() {
  auto analysis_pred = predictor_->Clone();
  std::unique_ptr<Predictor> pred(new Predictor(std::move(analysis_pred)));
  return pred;
}

void Predictor::ClearIntermediateTensor() {
  predictor_->ClearIntermediateTensor();
}

int PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    case PaddleDType::INT32:
      return sizeof(int32_t);
    case PaddleDType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

std::string GetPaddleVersion() { return paddle::get_version(); }

std::string UpdateDllFlag(const char *name, const char *value) {
  return paddle::UpdateDllFlag(name, value);
}

}  // namespace paddle_infer

namespace paddle_infer {
std::shared_ptr<Predictor> CreatePredictor(Config &config) {  // NOLINT
  std::shared_ptr<Predictor> predictor(new Predictor(config));
  return predictor;
}

PredictorPool::PredictorPool(const Config &config, size_t size) {
  PADDLE_ENFORCE_GE(
      size, 1UL,
      paddle::platform::errors::InvalidArgument(
          "The predictor pool size should be greater than 1, but it's (%d)",
          size));
  const paddle::AnalysisConfig &analysis_config =
      const_cast<Config *>(&config)->get_analysis_config();
  main_pred_.reset(new Predictor(config));
  for (size_t i = 0; i < size - 1; i++) {
    if (config.tensorrt_engine_enabled()) {
      paddle::AnalysisConfig config_tmp(analysis_config);
      preds_.emplace_back(std::unique_ptr<Predictor>(
          new Predictor(paddle::CreatePaddlePredictor(config_tmp))));
    } else {
      preds_.emplace_back(std::move(main_pred_->Clone()));
    }
  }
}

Predictor *PredictorPool::Retrive(size_t idx) {
  PADDLE_ENFORCE_LT(
      idx, preds_.size() + 1,
      paddle::platform::errors::InvalidArgument(
          "There are (%d) predictors in the pool, but the idx is (%d)", idx,
          preds_.size() + 1));
  if (idx == 0) {
    return main_pred_.get();
  }
  return preds_[idx - 1].get();
}
}  // namespace paddle_infer
