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
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#endif

#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif

#if PADDLE_WITH_ANAKIN
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#endif

DECLARE_bool(profile);

namespace paddle {

using inference::Singleton;
#if PADDLE_WITH_TENSORRT
using inference::tensorrt::TRTInt8Calibrator;
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
#endif

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}
}  // namespace

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";
    auto tracking_device = config_.use_gpu() ? platform::ProfilerState::kAll
                                             : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (!PrepareScope(parent_scope)) {
    return false;
  }
  if (!CreateExecutor()) {
    return false;
  }
  if (!PrepareProgram(program)) {
    return false;
  }

  // Prepare executor, create local variables.
  if (!PrepareExecutor()) {
    return true;
  }

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();

  return true;
}

bool AnalysisPredictor::PrepareScope(
    const std::shared_ptr<framework::Scope> &parent_scope) {
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        "Both program and parent_scope should be set in Clone mode.");
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
    status_is_cloned_ = false;
  }
  sub_scope_ = &scope_->NewScope();
  return true;
}
bool AnalysisPredictor::PrepareProgram(
    const std::shared_ptr<framework::ProgramDesc> &program) {
  if (!program) {
    if (!LoadProgramDesc()) return false;

    // If not cloned, the parameters should be loaded.
    // If config_.ir_optim() is True, parameters is loaded in
    // OptimizeInferenceProgram(), but other persistable variables
    // (like RAW type var) are not created in scope.
    // If config_.ir_optim() is False, parameters is loaded in LoadParameters(),
    // still need to create other persistable variables.
    // So in both case, create persistable variables at first.
    executor_->CreateVariables(*inference_program_, 0, true, sub_scope_);

    // Optimize the program, and load parameters and modify them in the
    // scope_.
    // This will change the scope_ address.
    if (config_.ir_optim()) {
      status_ir_optim_enabled_ = true;
      OptimizeInferenceProgram();
    } else {
      // Load parameters
      LOG(INFO) << "load parameters ";
      LoadParameters();
    }
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
  }

  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);

  return true;
}
bool AnalysisPredictor::CreateExecutor() {
  if (config_.use_gpu_) {
    status_use_gpu_ = true;
    place_ = paddle::platform::CUDAPlace(config_.device_id_);
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  executor_.reset(new paddle::framework::NaiveExecutor(place_));
  return true;
}
bool AnalysisPredictor::PrepareExecutor() {
  executor_->Prepare(sub_scope_, *inference_program_, 0,
                     config_.use_feed_fetch_ops_);

  PADDLE_ENFORCE_NOT_NULL(sub_scope_);

  return true;
}

void AnalysisPredictor::SetMkldnnThreadID(int tid) {
#ifdef PADDLE_WITH_MKLDNN
  platform::set_cur_thread_id(tid);
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
#endif
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
  VLOG(3) << "Predictor::predict";
  inference::Timer timer;
  timer.tic();
  // set feed variable
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

  // Collect variable shapes for memory optimization.
  if (need_collect_var_shapes_for_memory_optim()) {
    CollectVarShapes();
  }

  VLOG(3) << "predict cost: " << timer.toc() << "ms";

  // All the containers in the scope will be hold in inference, but the
  // operators assume that the container will be reset after each batch.
  // Here is a bugfix, collect all the container variables, and reset then to a
  // bool; the next time, the operator will call MutableData and construct a new
  // container again, so that the container will be empty for each batch.
  tensor_array_batch_cleaner_.CollectNoTensorVars(sub_scope_);
  tensor_array_batch_cleaner_.ResetNoTensorVars();
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
      auto dst_gpu_place = boost::get<platform::CUDAPlace>(place_);
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
    if (config_.specify_input_name_) {
      auto name = inputs[i].name;
      if (feed_names_.find(name) == feed_names_.end()) {
        LOG(ERROR) << "feed names from program do not have name: [" << name
                   << "] from specified input";
      }
      idx = feed_names_[name];
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
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    int idx = boost::get<int>(fetches_[i]->GetAttr("col"));
    PADDLE_ENFORCE((size_t)idx == i);
    framework::LoDTensor &fetch =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    output->name = fetches_[idx]->Input("X")[0];
    if (type == framework::proto::VarType::FP32) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == framework::proto::VarType::INT64) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else if (type == framework::proto::VarType::INT32) {
      GetFetchOne<int32_t>(fetch, output);
      output->dtype = PaddleDType::INT32;
    } else {
      LOG(ERROR) << "unknown type, only support float32, int64 and int32 now.";
    }
  }
  return true;
}

void AnalysisPredictor::PrepareArgument() {
  argument_.SetUseGPU(config_.use_gpu());
  argument_.SetGPUDeviceId(config_.gpu_device_id());
  argument_.SetEnableMemoryOptim(config_.enable_memory_optim());
  argument_.SetStaticMemoryOptim(config_.static_memory_optim_);
  argument_.SetStaticMemoryOptimForceUpdate(
      config_.static_memory_optim_force_update_);
  argument_.SetModelFromMemory(config_.model_from_memory_);
  argument_.SetEngineOptInfo(config_.engine_opt_info_);
  // Analyze inference_program
  argument_.SetUseAnakin(config_.anakin_engine_enabled());
  argument_.SetPredictorID(predictor_id_);
  if (!config_.model_dir().empty()) {
    argument_.SetModelDir(config_.model_dir());
  } else {
    PADDLE_ENFORCE(
        !config_.params_file().empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file().empty());
    std::string dir = inference::analysis::GetDirRoot(config_.prog_file());

    argument_.SetModelProgramPath(config_.prog_file());
    argument_.SetModelParamsPath(config_.params_file());
  }

  if (config_.use_gpu() && config_.tensorrt_engine_enabled()) {
    LOG(INFO) << "TensorRT subgraph engine is enabled";
    argument_.SetUseTensorRT(true);
    argument_.SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_.SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
    argument_.SetTensorRtMinSubgraphSize(config_.tensorrt_min_subgraph_size_);
    argument_.SetTensorRtPrecisionMode(config_.tensorrt_precision_mode_);
    argument_.SetTensorRtUseStaticEngine(config_.trt_use_static_engine_);
  }

  if (config_.use_gpu() && config_.anakin_engine_enabled()) {
    argument_.SetAnakinMaxBatchSize(config_.anakin_max_batchsize_);
    argument_.SetAnakinMaxInputShape(config_.anakin_max_input_shape_);
    argument_.SetAnakinMinSubgraphSize(config_.anakin_min_subgraph_size_);
    LOG(INFO) << "Anakin subgraph engine is enabled";
  }

  if (config_.use_mkldnn_) {
    LOG(INFO) << "MKLDNN is enabled";
    argument_.SetMKLDNNEnabledOpTypes(config_.mkldnn_enabled_op_types_);
  }

#ifdef PADDLE_WITH_MKLDNN
  if (config_.mkldnn_quantizer_enabled()) {
    LOG(INFO) << "Quantization is enabled";
    argument_.SetQuantizeEnabledOpTypes(
        config_.mkldnn_quantizer_config()->enabled_op_types());
    argument_.SetQuantizeExcludedOpIds(
        config_.mkldnn_quantizer_config()->excluded_op_ids());
  }
#endif

  auto passes = config_.pass_builder()->AllPasses();
  if (!config_.ir_optim()) {
    passes.clear();
    LOG(INFO) << "ir_optim is turned off, no IR pass will be executed";
  }
  argument_.SetIrAnalysisPasses(passes);
  argument_.SetAnalysisPasses(config_.pass_builder()->AnalysisPasses());
  argument_.SetScopeNotOwned(scope_.get());
}

// NOTE All the members in AnalysisConfig should be copied to Argument.
void AnalysisPredictor::OptimizeInferenceProgram() {
  status_program_optimized_ = true;

  PrepareArgument();
  Analyzer().Run(&argument_);

  PADDLE_ENFORCE(argument_.scope_valid());
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((&argument_), ir_analyzed_program);
  inference_program_.reset(
      new framework::ProgramDesc(argument_.ir_analyzed_program()));
  LOG(INFO) << "== optimize end ==";
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    AnalysisConfig, PaddleEngineKind::kAnalysis>(const AnalysisConfig &config) {
  VLOG(3) << "create AnalysisConfig";
  if (config.use_gpu()) {
    // 1. GPU memory
    PADDLE_ENFORCE_GE(config.memory_pool_init_size_mb(), 0.f);
    PADDLE_ENFORCE_GE(config.gpu_device_id(), 0, "Invalid device id %d",
                      config.gpu_device_id());
    std::vector<std::string> flags;

    float fraction_of_gpu_memory = config.fraction_of_gpu_memory_for_pool();
    if (fraction_of_gpu_memory > 0.95f) {
      LOG(ERROR)
          << "Allocate too much memory for the GPU memory pool, assigned "
          << config.memory_pool_init_size_mb() << " MB";
      LOG(ERROR)
          << "Try to shink the value by setting AnalysisConfig::EnableGpu(...)";
    }

    if (fraction_of_gpu_memory >= 0.0f || fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  auto predictor_p = dynamic_cast<AnalysisPredictor *>(predictor.get());

  if (!predictor_p->Init(nullptr)) {
    return nullptr;
  }

  if (config.mkldnn_quantizer_enabled() && !predictor_p->MkldnnQuantize()) {
    return nullptr;
  }

  return predictor;
}

bool AnalysisPredictor::MkldnnQuantize() {
#if PADDLE_WITH_MKLDNN
  if (!mkldnn_quantizer_)
    mkldnn_quantizer_ = new AnalysisPredictor::MkldnnQuantizer(
        *this, config_.mkldnn_quantizer_config());
  return mkldnn_quantizer_->Quantize();
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnQuantizer";
  return false;
#endif
}

void AnalysisPredictor::PrepareFeedFetch() {
  PADDLE_ENFORCE_NOT_NULL(sub_scope_);
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx2feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == "fetch") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      idx2fetches_[idx] = op->Input("X")[0];
    }
  }
}

void AnalysisPredictor::CreateFeedFetchVar(framework::Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope);
  auto *var = scope->Var("feed");
  var->GetMutable<framework::FeedFetchList>();
  var = scope->Var("fetch");
  var->GetMutable<framework::FeedFetchList>();
}

std::vector<std::string> AnalysisPredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : idx2feeds_) {
    input_names.push_back(item.second);
  }
  return input_names;
}

std::vector<std::string> AnalysisPredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : idx2fetches_) {
    output_names.push_back(item.second);
  }
  return output_names;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = boost::get<platform::CUDAPlace>(place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }

  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {
    res->SetPlace(PaddlePlace::kCPU);
  } else {
    auto gpu_place = boost::get<platform::CUDAPlace>(place_);
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

bool AnalysisPredictor::ZeroCopyRun() {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
  executor_->Run();
  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();
  return true;
}

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty() && !config_.params_file().empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file();
  } else {
    if (config_.model_dir().empty() && config_.prog_file().empty()) {
      LOG(ERROR)
          << "Either model_dir or (prog_file, param_file) should be set.";
      return false;
    }
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.", config_.model_dir(),
        config_.params_file());
    return false;
  }

  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE(static_cast<bool>(fin.is_open()), "Cannot open file %s",
                   filename);
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  inference_program_.reset(new framework::ProgramDesc(proto));
  return true;
}

bool AnalysisPredictor::LoadParameters() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          "The inference program should be loaded first.");

  const auto &global_block = inference_program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();

      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      if (!config_.params_file().empty()) {
        params.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc *op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {config_.model_dir() + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!config_.params_file().empty()) {
    // sort paramlist to have consistent ordering
    std::sort(params.begin(), params.end());
    // append just the load_combine op
    framework::OpDesc *op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", params);
    op->SetAttr("file_path", {config_.params_file()});
    op->CheckAttrs();
  }

  // Use NaiveExecutor to Load parameters.
  framework::NaiveExecutor e(place_);
  e.Prepare(scope_.get(), *load_program, 0, false);
  e.Run();
  VLOG(3) << "get " << scope_->LocalVarNames().size() << " vars after load";

  return true;
}

#if PADDLE_WITH_TENSORRT
bool AnalysisPredictor::SaveTrtCalibToDisk() {
  PADDLE_ENFORCE(config_.tensorrt_engine_enabled(),
                 "This func can be invoked only in trt mode");
  auto &block = inference_program_->Block(0);
  for (auto &op_desc : block.AllOps()) {
    if (op_desc->Type() == "tensorrt_engine") {
      std::string engine_name =
          boost::get<std::string>(op_desc->GetAttr("engine_key"));
      if (!Singleton<TRTCalibratorEngineManager>::Global().Has(engine_name)) {
        LOG(ERROR) << "You should run the predictor(with trt) on the real data "
                      "to generate calibration info";
        return false;
      }
      TRTCalibratorEngine *calib_engine =
          Singleton<TRTCalibratorEngineManager>::Global().Get(engine_name);
      LOG(INFO) << "Wait for calib threads done.";
      calib_engine->calib_->waitAndSetDone();
      LOG(INFO) << "Generating TRT Calibration table data, this may cost a lot "
                   "of time...";
      calib_engine->thr_->join();
      std::string calibration_table_data =
          calib_engine->calib_->getCalibrationTableAsString();

      if (calibration_table_data.empty()) {
        LOG(ERROR) << "the calibration table is empty.";
        return false;
      }

      std::string model_opt_cache_dir =
          argument_.Has("model_dir")
              ? argument_.model_dir()
              : inference::analysis::GetDirRoot(argument_.model_program_path());

      std::string calibration_table_data_path =
          inference::analysis::GetTrtCalibPath(
              inference::analysis::GetOrCreateModelOptCacheDir(
                  model_opt_cache_dir),
              engine_name);

      std::ofstream ofile(calibration_table_data_path, std::ios::out);
      LOG(INFO) << "Write Paddle-TRT INT8 calibration table data to file "
                << calibration_table_data_path;
      ofile << calibration_table_data;
      ofile.close();
    }
  }
  // Free all calibrator resources.
  Singleton<TRTCalibratorEngineManager>::Global().DeleteALL();
  return true;
}
#endif

AnalysisPredictor::~AnalysisPredictor() {
#if PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled() &&
      config_.tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8 &&
      Singleton<TRTCalibratorEngineManager>::Global().Has()) {
    SaveTrtCalibToDisk();
  }
#endif
  if (FLAGS_profile) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }

#if PADDLE_WITH_MKLDNN
  if (mkldnn_quantizer_) {
    delete mkldnn_quantizer_;
    mkldnn_quantizer_ = nullptr;
  }
#endif

  // TODO(Superjomn) deduce the directory path.
  std::string out_path = inference::analysis::GetMemoryCachePath(
      config_.model_dir(), config_.prog_file());
  if (need_collect_var_shapes_for_memory_optim()) {
    SerializeBatchVarShapes(out_path);
  }
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone() {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new AnalysisPredictor(config_);
  x->Init(scope_, inference_program_);
  return std::unique_ptr<PaddlePredictor>(x);
}

void AnalysisPredictor::CollectVarShapes() {
  VLOG(4) << "Collecting var shapes";
  if (batch_var_shapes_.size() >= max_shape_collect_count_) return;
  std::map<std::string, std::vector<int>> var_shapes;
  for (auto var_name : inference_program_->Block(0).LocalVarNames()) {
    auto *var = sub_scope_->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var);
    if (var->Type() == framework::VarTypeTrait<framework::LoDTensor>::kId ||
        var->Type() == framework::VarTypeTrait<framework::Tensor>::kId) {
      auto &tensor = var->Get<framework::LoDTensor>();
      auto shape = framework::vectorize(tensor.dims());
      var_shapes[var_name].assign(shape.begin(), shape.end());
    }
  }
  batch_var_shapes_.push_back(var_shapes);
  LOG_FIRST_N(INFO, 1) << "Collected " << batch_var_shapes_.size()
                       << " batch of var shapes for analysis";
}

void AnalysisPredictor::SerializeBatchVarShapes(const std::string &path) {
  LOG(INFO) << "serialize batch var shapes to " << path;
  std::ofstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "failed to serialize the var shapes to " << path;
    return;
  }

  // The sirialized data format:
  // <tensor_name>:dim0,dim1,dim2,;
  for (auto &batch : batch_var_shapes_) {
    for (auto &ele : batch) {
      file << ele.first << ":";
      for (size_t i = 0; i < ele.second.size() - 1; i++) {
        file << ele.second[i] << ",";
      }
      file << ele.second.back() << ";";
    }
    file << "\n";
  }
}

bool AnalysisPredictor::need_collect_var_shapes_for_memory_optim() {
  if (need_collect_var_shapes_ >= 0) return need_collect_var_shapes_;
  bool need = false;
  // check if the cache exists
  if (!config_.enable_memory_optim()) {
    need = false;
  } else if (config_.static_memory_optim_ &&
             !inference::IsFileExists(inference::analysis::GetMemoryCachePath(
                 config_.model_dir(), config_.prog_file()))) {
    need = true;
  } else if (config_.static_memory_optim_ &&
             config_.static_memory_optim_force_update_) {
    need = true;
  }

  need_collect_var_shapes_ = need ? 1 : 0;
  return need;
}

std::string AnalysisPredictor::GetSerializedProgram() const {
  return inference_program_->Proto()->SerializeAsString();
}

// Add SaveOptimModel
void AnalysisPredictor::SaveOptimModel(const std::string &dir) {
  // save model
  std::string model_name = dir + "/model";
  std::ofstream outfile;
  outfile.open(model_name, std::ios::out | std::ios::binary);
  std::string inference_prog_desc = GetSerializedProgram();
  outfile << inference_prog_desc;
  // save params
  framework::ProgramDesc save_program;
  auto *save_block = save_program.MutableBlock(0);

  const framework::ProgramDesc &main_program = program();
  const framework::BlockDesc &global_block = main_program.Block(0);
  std::vector<std::string> save_var_list;
  for (framework::VarDesc *var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      framework::VarDesc *new_var = save_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      save_var_list.push_back(new_var->Name());
    }
  }
  std::sort(save_var_list.begin(), save_var_list.end());
  auto *op = save_block->AppendOp();
  op->SetType("save_combine");
  op->SetInput("X", save_var_list);
  op->SetAttr("file_path", dir + "/params");
  op->CheckAttrs();

  platform::CPUPlace place;
  framework::Executor exe(place);
  exe.Run(save_program, scope(), 0, true, true);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<AnalysisConfig>(
    const AnalysisConfig &config) {
  return CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
      config);
}

}  // namespace paddle

#if PADDLE_WITH_TENSORRT
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
USE_TRT_CONVERTER(split);
USE_TRT_CONVERTER(prelu);
USE_TRT_CONVERTER(conv2d_transpose);
USE_TRT_CONVERTER(leaky_relu);
#endif

#if PADDLE_WITH_ANAKIN
USE_ANAKIN_CONVERTER(mul);
USE_ANAKIN_CONVERTER(fc);
USE_ANAKIN_CONVERTER(conv2d);
USE_ANAKIN_CONVERTER(conv2d_fusion);
USE_ANAKIN_CONVERTER(concat);
USE_ANAKIN_CONVERTER(split);
USE_ANAKIN_CONVERTER(relu);
USE_ANAKIN_CONVERTER(sigmoid);
USE_ANAKIN_CONVERTER(tanh);
USE_ANAKIN_CONVERTER(pool2d);
USE_ANAKIN_CONVERTER(elementwise_add);
USE_ANAKIN_CONVERTER(elementwise_mul);
USE_ANAKIN_CONVERTER(batch_norm);
USE_ANAKIN_CONVERTER(flatten);
USE_ANAKIN_CONVERTER(reshape);
USE_ANAKIN_CONVERTER(transpose);
USE_ANAKIN_CONVERTER(softmax);
USE_ANAKIN_CONVERTER(detection_out);
USE_ANAKIN_CONVERTER(density_prior_box);
USE_ANAKIN_CONVERTER(dropout);
USE_ANAKIN_CONVERTER(sum);
USE_ANAKIN_CONVERTER(prior_box);
#endif
