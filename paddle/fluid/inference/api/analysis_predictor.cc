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
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid//platform/device/gpu/gpu_types.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/pass_result_info.h"
#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/infer_context.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/resource_manager.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/inference/utils/model_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/utils/string/split.h"

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/phi/backends/dynload/mklml.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#endif

#ifdef PADDLE_WITH_ONNXRUNTIME
#include "paddle/fluid/inference/api/onnxruntime_predictor.h"
#endif

#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/paddle_ipu_handler.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_info.h"
#endif

namespace paddle {
namespace {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void UpdatePrivateDeviceContext(InferGPUContext *gpu_context,
                                GPUContextResource *gpu_resource,
                                Place place_) {
  gpu_context->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                                .GetAllocator(place_, gpu_resource->GetStream())
                                .get());
  gpu_context->SetPinnedAllocator(
      memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
  gpu_context->SetHostAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetAllocator(platform::CPUPlace())
                                    .get());
  gpu_context->SetZeroAllocator(memory::allocation::AllocatorFacade::Instance()
                                    .GetZeroAllocator(place_)
                                    .get());
  gpu_context->SetHostZeroAllocator(
      memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(platform::CPUPlace())
          .get());
  gpu_context->SetGenerator(
      phi::DefaultCUDAGenerator(place_.GetDeviceId()).get());
  gpu_context->SetHostGenerator(phi::DefaultCPUGenerator().get());

  gpu_context->SetStream(gpu_resource->GetStream());
  gpu_context->SetBlasHandle(gpu_resource->GetBlasHandleCreator());
  gpu_context->SetBlasTensorCoreHandle(
      gpu_resource->GetBlasTensorCoreHandleCreator());
  gpu_context->SetBlasTF32Handle(
      gpu_resource->GetBlasTF32TensorCoreHandleCreator());
  gpu_context->SetDnnHandle(gpu_resource->GetDnnHandleCreator());
  gpu_context->SetSolverHandle(gpu_resource->GetSolverDnHandleCreator());
  gpu_context->SetSparseHandle(gpu_resource->GetSparseHandleCreator());
  gpu_context->SetEigenDevice(gpu_resource->GetGpuEigenDevice());

  gpu_context->SetComputeCapability(gpu_resource->GetGpuComputeCapability());
  gpu_context->SetMaxThreadsPerBlock(gpu_resource->GetGpuMaxThreadsPerBlock());
  gpu_context->SetMaxThreadsPerMultiProcessor(
      gpu_resource->GetGpuMaxThreadsPerMp());
  gpu_context->SetMaxGridDimSize(gpu_resource->GetGpuMaxGridDimSize());
  gpu_context->SetMultiProcessors(gpu_resource->GetGPUMultiProcessors());
  gpu_context->SetDriverVersion(gpu_resource->GetGpuDriverVersion());
  gpu_context->SetRuntimeVersion(gpu_resource->GetGpuRuntimeVersion());
  VLOG(1) << "thread id is " << std::this_thread::get_id() << ", stream id is "
          << reinterpret_cast<void *>(gpu_resource->GetStream())
          << ", allotor ptr is "
          << reinterpret_cast<void *>(
                 memory::allocation::AllocatorFacade::Instance()
                     .GetAllocator(place_, gpu_resource->GetStream())
                     .get());
}
#endif
}  // namespace

#ifdef PADDLE_WITH_TENSORRT
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;
using inference::tensorrt::TRTInt8Calibrator;
#endif

int AnalysisPredictor::clone_num_ = 1;

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

phi::DataType ConvertPrecision(AnalysisConfig::Precision precision) {
  switch (precision) {
    case AnalysisConfig::Precision::kFloat32:
      return phi::DataType::FLOAT32;
    case AnalysisConfig::Precision::kHalf:
      return phi::DataType::FLOAT16;
    case AnalysisConfig::Precision::kBf16:
      return phi::DataType::BFLOAT16;
    case AnalysisConfig::Precision::kInt8:
      return phi::DataType::INT8;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Paddle Inference not support precision. We now only support "
          "Float32, Half, Bfloat16 and Int8"));
      return phi::DataType::FLOAT32;
  }
}

phi::Backend ConvertBackend(paddle_infer::PlaceType backend) {
  switch (backend) {
    case paddle_infer::PlaceType::kGPU:
      // NOTE: phi also support phi::Backend::GPUDNN.
      return phi::Backend::GPU;
    case paddle_infer::PlaceType::kXPU:
      return phi::Backend::XPU;
    case paddle_infer::PlaceType::kCPU:
      return phi::Backend::CPU;
    case paddle_infer::PlaceType::kIPU:
      return phi::Backend::IPU;
    case paddle_infer::PlaceType::kCUSTOM:
      return phi::Backend::CUSTOM;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Paddle Inference not support backend, we now only support GPU, XPU "
          "and CPU."));
      return phi::Backend::CPU;
  }
}

bool PaddleTensorToDenseTensor(const PaddleTensor &pt,
                               phi::DenseTensor *t,
                               const platform::Place &place) {
  framework::DDim ddim = phi::make_ddim(pt.shape);
  void *input_ptr;
  if (pt.dtype == PaddleDType::INT64) {
    input_ptr = t->mutable_data<int64_t>(ddim, place);
  } else if (pt.dtype == PaddleDType::FLOAT32) {
    input_ptr = t->mutable_data<float>(ddim, place);
  } else if (pt.dtype == PaddleDType::INT32) {
    input_ptr = t->mutable_data<int32_t>(ddim, place);
  } else if (pt.dtype == PaddleDType::FLOAT16) {
    input_ptr = t->mutable_data<float16>(ddim, place);
  } else if (pt.dtype == PaddleDType::BFLOAT16) {
    input_ptr = t->mutable_data<bfloat16>(ddim, place);
  } else {
    LOG(ERROR) << "unsupported feed type " << pt.dtype;
    return false;
  }
  // NOTE(Aurelius84): Some kernels support zero shape input
  // without memory holder, we should skip enforce logic.
  bool has_zero_dim = (phi::product(ddim) == 0);
  VLOG(3) << "Found zero dim: " << has_zero_dim
          << " from input with ddim: " << ddim;
  if (!has_zero_dim) {
    PADDLE_ENFORCE_NOT_NULL(
        input_ptr,
        paddle::platform::errors::Fatal(
            "Cannot convert to LoDTensor because LoDTensor creation failed."));
    PADDLE_ENFORCE_NOT_NULL(
        pt.data.data(),
        paddle::platform::errors::InvalidArgument(
            "The data contained in the input PaddleTensor is illegal."));
    PADDLE_ENFORCE_EQ(
        pt.data.length(),
        t->numel() * phi::SizeOf(t->dtype()),
        paddle::platform::errors::InvalidArgument(
            "The data contained in the input PaddleTensor had wrong length."));
  }

  if (platform::is_cpu_place(place)) {
    // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
    if (input_ptr != nullptr) {
      std::memcpy(
          static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
    }
  } else if (platform::is_ipu_place(place)) {
#ifdef PADDLE_WITH_IPU
    std::memcpy(
        static_cast<void *>(input_ptr), pt.data.data(), pt.data.length());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with WITH_IPU, should not reach here."));
#endif
  } else if (platform::is_gpu_place(place)) {
    PADDLE_ENFORCE_EQ(platform::is_xpu_place(place),
                      false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto *dev_ctx = static_cast<const phi::GPUContext *>(pool.Get(place));
    auto dst_gpu_place = place;
    memory::Copy(dst_gpu_place,
                 static_cast<void *>(input_ptr),
                 platform::CPUPlace(),
                 pt.data.data(),
                 pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with CUDA, should not reach here."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto dst_xpu_place = place;
    memory::Copy(dst_xpu_place,
                 static_cast<void *>(input_ptr),
                 platform::CPUPlace(),
                 pt.data.data(),
                 pt.data.length());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with XPU, should not reach here."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    paddle::platform::DeviceContextPool &pool =
        paddle::platform::DeviceContextPool::Instance();
    auto custom_place = place;
    auto *dev_ctx = static_cast<const paddle::platform::CustomDeviceContext *>(
        pool.Get(custom_place));
    memory::Copy(custom_place,
                 static_cast<void *>(input_ptr),
                 platform::CPUPlace(),
                 pt.data.data(),
                 pt.data.length(),
                 dev_ctx->stream());
#else
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Not compile with CUSTOM_DEVICE, should not reach here."));
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU, XPU and CUSTOM_DEVICE "
        "now."));
  }
  // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
  framework::LoD lod;
  for (auto &level : pt.lod) {
    lod.emplace_back(level);
  }
  t->set_lod(lod);
  return true;
}
}  // namespace

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";
  if (config_.with_profile_) {
    LOG(WARNING) << "Profiler is activated, which might affect the performance";
    auto tracking_device = config_.use_gpu() ? platform::ProfilerState::kAll
                                             : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  } else {
    VLOG(2) << "Profiler is deactivated, and no profiling report will be "
               "generated.";
  }

  if (!status_is_cloned_) {
    root_predictor_id_ = predictor_id_;
  }

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());

  if (!PrepareScope(parent_scope)) {
    return false;
  }

  InitPlace();

  if (!CreateExecutor()) {
    return false;
  }
  if (!PrepareProgram(program)) {
    return false;
  }

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();

  // Prepare executor, create local variables.
  if (!PrepareExecutor()) {
    return true;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // TODO(inference): Now only gpu with external stream support private
  // device_context.
  if (config_.use_gpu_ && config_.use_external_stream_) {
    private_context_ = true;
  }
  if (private_context_) {
    if (!status_is_cloned_) {
      predictor_stream_ = config_.GetExecStream();
    }
    // NOTE: If the external_stream equals to global_device_contexts's stream,
    // then fallback.
    auto global_stream =
        static_cast<phi::GPUContext *>(
            platform::DeviceContextPool::Instance().Get(place_))
            ->stream();
    if (predictor_stream_ != global_stream) {
      InitResourceManager(predictor_stream_);
      InitDeviceContexts();
    }
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (config_.use_xpu_ && !config_.use_lite_) {
    private_context_ = true;
    if (!status_is_cloned_ && config_.external_stream_enabled()) {
      predictor_stream_ = config_.GetExecStream();
    }
    if (predictor_stream_ == nullptr) {
      auto *global_context = static_cast<phi::XPUContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
      predictor_stream_ = global_context->stream();
    }
    InitDeviceContexts();
  }
#endif

  inference::DisplayMemoryInfo(place_, "Init predictor");
  return true;
}

void AnalysisPredictor::InitPlace() {
  if (config_.use_gpu()) {
    PADDLE_ENFORCE_EQ(config_.use_xpu(),
                      false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    place_ = paddle::platform::CUDAPlace(config_.gpu_device_id());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (config_.thread_local_stream_enabled()) {
      LOG_FIRST_N(WARNING, 1) << "We will remove this interface in the future. "
                                 "Please use config.SetExecStream instead.";
    }
#endif
  } else if (config_.use_xpu()) {
    if (config_.lite_engine_enabled()) {
#ifdef LITE_SUBGRAPH_WITH_XPU
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of Host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      place_ = paddle::platform::CPUPlace();
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use an XPU lite engine, but Paddle was not compiled "
          "with it."));
#endif  // LITE_SUBGRAPH_WITH_XPU
    } else {
#ifdef PADDLE_WITH_XPU
      phi::backends::xpu::SetXPUDeviceId(config_.xpu_device_id());
      place_ = paddle::platform::XPUPlace(config_.xpu_device_id());
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "You tried to use XPU forward propagation (inference without lite "
          "engine), but Paddle was not compiled "
          "with WITH_XPU."));
#endif  // PADDLE_WITH_XPU
    }
  } else if (config_.NNAdapter().use_nnadapter) {
    if (config_.lite_engine_enabled()) {
      place_ = paddle::platform::CPUPlace();
#ifndef LITE_SUBGRAPH_WITH_NNADAPTER
      PADDLE_THROW(
          platform::errors::Unavailable("You tried to use an NNAdapter lite "
                                        "engine, but Paddle was not compiled "
                                        "with it."));
#endif  // LITE_SUBGRAPH_WITH_NNADAPTER
    } else {
      PADDLE_THROW(
          platform::errors::Unavailable("You tried to use NNadapter forward "
                                        "propagation (inference without lite "
                                        "engine), but Paddle was not compiled "
                                        "with LITE_WITH_NNADAPTER."));
    }
  } else if (config_.use_ipu()) {
#ifdef PADDLE_WITH_IPU
    place_ = paddle::platform::IPUPlace();
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to use IPU forward propagation, but Paddle was not compiled "
        "with WITH_IPU."));
#endif
  } else if (config_.use_custom_device()) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    place_ = paddle::platform::CustomPlace(config_.custom_device_type(),
                                           config_.custom_device_id());
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to use CustomDevice forward propagation, but Paddle was not "
        "compiled "
        "with WITH_CUSTOM_DEVICE."));
#endif
  } else {
    place_ = paddle::platform::CPUPlace();
  }
}

void AnalysisPredictor::InitResourceManager(void *stream) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  predictor_stream_ =
      ResourceManager::Instance().InitGPUResource(place_, stream);
#endif
}

void AnalysisPredictor::InitDeviceContexts() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Init GPUContext.
  if (place_.GetType() == phi::AllocationType::GPU) {
    device_contexts_.emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto *gpu_resource =
              ResourceManager::Instance().GetGPUResource(predictor_stream_);
          auto *gpu_context = new InferGPUContext(place_);
          UpdatePrivateDeviceContext(gpu_context, gpu_resource, place_);
          return std::unique_ptr<phi::DeviceContext>(gpu_context);
        }));
  }
#endif
#ifdef PADDLE_WITH_XPU
  if (place_.GetType() == phi::AllocationType::XPU) {
    device_contexts_.emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto &instance = memory::allocation::AllocatorFacade::Instance();
          auto *xpu_context =
              new InferXPUContext(place_, config_.xpu_config().context_gm_size);
          xpu_context->SetAllocator(instance.GetAllocator(place_).get());
          xpu_context->SetGenerator(
              phi::DefaultXPUGenerator(place_.GetDeviceId()).get());
          xpu_context->SetHostAllocator(
              instance.GetAllocator(platform::CPUPlace()).get());
          xpu_context->SetHostGenerator(phi::DefaultCPUGenerator().get());
          xpu_context->SetZeroAllocator(
              instance.GetZeroAllocator(place_).get());
          xpu_context->SetHostZeroAllocator(
              instance.GetZeroAllocator(platform::CPUPlace()).get());
          xpu_context->SetStream(predictor_stream_);
          return std::unique_ptr<phi::DeviceContext>(xpu_context);
        }));
  }
#endif
}

void *AnalysisPredictor::GetExecStream() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place_.GetType() == phi::AllocationType::GPU) {
    if (private_context_) {
      return predictor_stream_;
    } else {
      paddle::platform::DeviceContextPool &pool =
          paddle::platform::DeviceContextPool::Instance();
      return reinterpret_cast<const phi::GPUContext *>(pool.Get(place_))
          ->stream();
    }
  }
#endif
#if defined(PADDLE_WITH_XPU)
  if (place_.GetType() == phi::AllocationType::XPU) {
    if (private_context_) {
      return predictor_stream_;
    } else {
      paddle::platform::DeviceContextPool &pool =
          paddle::platform::DeviceContextPool::Instance();
      return reinterpret_cast<const phi::XPUContext *>(pool.Get(place_))
          ->stream();
    }
  }
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (place_.GetType() == phi::AllocationType::CUSTOM) {
    paddle::platform::DeviceContextPool &pool =
        paddle::platform::DeviceContextPool::Instance();
    return reinterpret_cast<const phi::CustomContext *>(pool.Get(place_))
        ->stream();
  }
#endif
  // TODO(inference): Support other backends.
  return nullptr;
}

const void *AnalysisPredictor::GetDeviceContexts() const {
  if (private_context_) {
    return &device_contexts_;
  } else {
    paddle::platform::DeviceContextPool &pool =
        paddle::platform::DeviceContextPool::Instance();
    const auto &dev_ctxs = pool.device_contexts();
    return &dev_ctxs;
  }
}

bool AnalysisPredictor::PrepareScope(
    const std::shared_ptr<framework::Scope> &parent_scope) {
#ifdef PADDLE_WITH_XPU
  // Set "XPU_PADDLE_L3_SIZE" to "0" to avoid malloc l3 cache when xpu_context
  // init.
  setenv("XPU_PADDLE_L3_SIZE", "0", 0);
#endif
  if (parent_scope) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        platform::errors::PreconditionNotMet(
            "Both program and parent_scope should be set in Clone mode."));
    scope_ = parent_scope;
    status_is_cloned_ = true;
  } else {
    paddle::framework::InitMemoryMethod();
    paddle::framework::InitDevices();
    paddle::framework::InitDefaultKernelSignatureMap();
    // TODO(wilber): we need to release memory occupied by weights.
    scope_ = std::make_unique<paddle::framework::Scope>();
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

    // if enable_ir_optim_ is false,
    // the analysis pass(op fuse, graph analysis, trt subgraph, mkldnn etc) will
    // not be executed.
    model_precision_ =
        paddle::inference::GetModelPrecision(*inference_program_);
    OptimizeInferenceProgram();
  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
    if (config_.apply_optim_) {
      VLOG(3)
          << "apply_optim is enabled, will call OptimizeInferenceProgram().";
      OptimizeInferenceProgram();
    }
  }
  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);
  return true;
}

bool AnalysisPredictor::CreateExecutor() {
  executor_ = std::make_unique<paddle::framework::NaiveExecutor>(place_);
  return true;
}

static bool IsPrepareDataOptTargetOp(framework::OpDesc *op) {
  // here is prepare data optimization related bad cases:
  // let's assume an op behind conditional_block and if conditional_block
  // chooses branch 1, the op need to call prepare data. else the op don't need
  // to call prepare data. In running, if predictor chooses branch 2, then
  // optimization takes effect, later issue is followed if predictor chooses
  // branch 1, because the op lost chance to prepare data.
  std::vector<std::string> op_type = {"conditional_block_infer",
                                      "select_input"};
  for (const auto &type : op_type) {
    if (op->Type() == type) {
      return true;
    }
  }
  return false;
}

static void DisablePrepareDataOpt(
    std::shared_ptr<framework::ProgramDesc> inference_program,
    int block,
    bool pre_disable_opt) {
  bool disable_opt = false;
  auto &infer_block = inference_program->Block(block);
  for (auto *op : infer_block.AllOps()) {
    if (disable_opt || pre_disable_opt) {
      op->SetAttr("inference_force_prepare_data", true);
    }
    if (op->HasAttr("sub_block")) {
      int blockID = op->GetBlockAttrId("sub_block");
      DisablePrepareDataOpt(
          inference_program, blockID, disable_opt || pre_disable_opt);
    }
    // disable prepare data if unfriendly op is found
    if (!disable_opt) {
      disable_opt = IsPrepareDataOptTargetOp(op);
    }
  }
}

bool AnalysisPredictor::PrepareExecutor() {
  if (config_.dist_config().use_dist_model()) {
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
    VLOG(3) << "use_dist_model is enabled, will init FleetExecutor.";
    return PrepareFleetExecutor();
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't use FleetExecutor since it's not compiled with PSCORE,"
        "Please recompile or reinstall Paddle with PSCORE support."));
#endif
  }
  DisablePrepareDataOpt(inference_program_, 0, false);

  executor_->Prepare(
      sub_scope_, *inference_program_, 0, config_.use_feed_fetch_ops_);

  if (config_.enable_memory_optim_) {
    auto *pass_res_info =
        inference::analysis::PassResultInfoForRuntime::Instance();
    auto reuse_table =
        pass_res_info->Get<std::unordered_map<std::string, std::string>>(
            root_predictor_id_, "memory_optimize_pass");
    executor_->MakeReusePlan(reuse_table);
  }

  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          platform::errors::PreconditionNotMet(
                              "The sub_scope should not be nullptr."));

  return true;
}

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
bool AnalysisPredictor::PrepareFleetExecutor() {
  VLOG(3) << "AnalysisPredictor::PrepareFleetExecutor()";
  if (config_.dist_config().nranks() > 1 && !CommInit()) {
    return false;
  }
  task_node_ = std::make_unique<distributed::TaskNode>(
      inference_program_.get(), config_.dist_config().rank());
  // With auto cut, there is no concept of pp, no need to add dependency.
  task_node_->SetType("Compute");
  task_node_->Init(config_.use_feed_fetch_ops_enabled());
  executor_desc_ = distributed::FleetExecutorDesc();
  executor_desc_.set_cur_rank(config_.dist_config().rank());
  std::unordered_map<int64_t, int64_t> id_to_rank;
  for (int i = 0; i < config_.dist_config().nranks(); ++i) {
    distributed::RankInfo *rank_info = executor_desc_.add_cluster_info();
    rank_info->set_rank(i);
    rank_info->set_ip_port(config_.dist_config().trainer_endpoints()[i]);
    id_to_rank.insert({i, i});
  }
  fleet_exe_ = std::make_unique<distributed::FleetExecutor>(executor_desc_);
  // NOTE: Vars of feed fetch ops are not persistable,
  // which will result in that those vars will be created in
  // the subscope (microscope) in fleet executor. This will
  // cause that the GetInputTensor/GetOutputTensor funct
  // in analysis predictor cannot find those vars in the scope
  // returned by the DistModel, since DistModel only return the
  // root scope. So, those vars must  to be created in the root
  // scope instead of in the microscope
  std::vector<std::string> feed_fetch_vars;
  for (auto pair : idx2feeds_) {
    feed_fetch_vars.emplace_back(pair.second);
  }
  for (auto pair : idx2fetches_) {
    feed_fetch_vars.emplace_back(pair.second);
  }
  fleet_exe_->Init(config_.dist_config().carrier_id(),
                   *(inference_program_.get()),
                   scope_.get(),
                   place_,
                   1,
                   {task_node_.get()},
                   id_to_rank,
                   feed_fetch_vars);
  return true;
}

bool AnalysisPredictor::CommInit() {
  std::map<int64_t, std::vector<int64_t>> ring_id_to_ranks{};
  std::map<int64_t, std::vector<int64_t>> rank_to_ring_ids{};
  if (!LoadConverterConfig(&ring_id_to_ranks, &rank_to_ring_ids)) {
    VLOG(3) << "Load converter config failed, DistModel init failed.";
    return false;
  }
  std::unique_ptr<framework::ProgramDesc> comm_init_program(
      new framework::ProgramDesc());
  framework::BlockDesc *comm_init_block = comm_init_program->MutableBlock(0);
  std::vector<int64_t> &ring_ids =
      rank_to_ring_ids[config_.dist_config().rank()];
  int64_t order = 0;
  std::string var_name_base = "comm_init_";
  for (int64_t ring_id : ring_ids) {
    VLOG(3) << "Init comm for ring id: " << ring_id;
    int64_t ranks_in_group = ring_id_to_ranks[ring_id].size();
    int64_t rank_in_group = 0;
    std::vector<int64_t> &ranks = ring_id_to_ranks[ring_id];
    for (int64_t rank : ranks) {
      if (config_.dist_config().rank() == rank) {
        break;
      }
      rank_in_group += 1;
    }
    std::vector<std::string> peer_endpoints;
    for (int64_t rank : ranks) {
      if (config_.dist_config().rank() == rank) {
        continue;
      }
      peer_endpoints.emplace_back(
          config_.dist_config().trainer_endpoints()[rank]);
    }
    InsertCommOp(var_name_base + std::to_string(order),
                 ranks_in_group,
                 rank_in_group,
                 peer_endpoints,
                 comm_init_block,
                 ring_id);
    order += 1;
  }
  framework::NaiveExecutor e(place_);
  e.CreateVariables(*comm_init_program, 0, true, scope_.get());
  e.Prepare(scope_.get(), *comm_init_program, 0, false);
  e.Run();
  VLOG(3) << "Comm init successful.";
  return true;
}

void AnalysisPredictor::InsertCommOp(
    std::string tmp_var_name,
    int nranks,
    int rank,
    const std::vector<std::string> &peer_endpoints,
    framework::BlockDesc *block,
    int ring_id) {
  /*
   * tmp_var_name: the var name for var comm_id
   * nranks: number of total ranks
   * rank: the rank of local rank in the comm group
   * peer_endpoints: peer's endpoints
   * block: the block where to insert the comm ops
   * ring_id: the ring_id to be inited
   */
  const std::string &endpoint = config_.dist_config().current_endpoint();
  std::stringstream ss;
  ss << "Init comm with tmp var: " << tmp_var_name
     << ". The ring id is: " << ring_id << ". The group has: " << nranks
     << " ranks. Current rank in the group is: " << rank
     << ". The endpoint is: " << endpoint << ". Peer endpoints are: ";
  for (auto ep : peer_endpoints) {
    ss << ep << ", ";
  }
  VLOG(3) << ss.str();
  std::string endpoints_str = config_.dist_config().current_endpoint();
  for (const auto &peer : peer_endpoints) {
    endpoints_str += "," + peer;
  }
  if (config_.use_gpu()) {
    framework::VarDesc *new_var = block->Var(tmp_var_name);
    new_var->SetType(framework::proto::VarType::RAW);
    new_var->SetPersistable(true);
    framework::OpDesc *gen_nccl_id_op = block->AppendOp();
    gen_nccl_id_op->SetType("c_gen_nccl_id");
    gen_nccl_id_op->SetOutput("Out", {tmp_var_name});
    gen_nccl_id_op->SetAttr("rank", rank);
    gen_nccl_id_op->SetAttr("endpoint",
                            config_.dist_config().current_endpoint());
    gen_nccl_id_op->SetAttr("other_endpoints", peer_endpoints);
    gen_nccl_id_op->SetAttr("ring_id", ring_id);
    gen_nccl_id_op->SetAttr("op_role",
                            static_cast<int>(framework::OpRole::kForward));
    gen_nccl_id_op->CheckAttrs();
    framework::OpDesc *comm_init_op = block->AppendOp();
    comm_init_op->SetType("c_comm_init");
    comm_init_op->SetInput("X", {tmp_var_name});
    comm_init_op->SetAttr("rank", rank);
    comm_init_op->SetAttr("nranks", nranks);
    comm_init_op->SetAttr("ring_id", ring_id);
    comm_init_op->SetAttr("endpoints", endpoints_str);
    comm_init_op->SetAttr("op_role",
                          static_cast<int>(framework::OpRole::kForward));
    comm_init_op->CheckAttrs();
  } else if (config_.use_xpu()) {
    framework::VarDesc *new_var = block->Var(tmp_var_name);
    new_var->SetType(framework::proto::VarType::RAW);
    new_var->SetPersistable(true);
    framework::OpDesc *gen_bkcl_id_op = block->AppendOp();
    gen_bkcl_id_op->SetType("c_gen_bkcl_id");
    gen_bkcl_id_op->SetOutput("Out", {tmp_var_name});
    gen_bkcl_id_op->SetAttr("rank", rank);
    gen_bkcl_id_op->SetAttr("endpoint",
                            config_.dist_config().current_endpoint());
    gen_bkcl_id_op->SetAttr("other_endpoints", peer_endpoints);
    gen_bkcl_id_op->SetAttr("ring_id", ring_id);
    gen_bkcl_id_op->SetAttr("op_role",
                            static_cast<int>(framework::OpRole::kForward));
    gen_bkcl_id_op->CheckAttrs();
    framework::OpDesc *comm_init_op = block->AppendOp();
    comm_init_op->SetType("c_comm_init");
    comm_init_op->SetInput("X", {tmp_var_name});
    comm_init_op->SetAttr("rank", rank);
    comm_init_op->SetAttr("nranks", nranks);
    comm_init_op->SetAttr("ring_id", ring_id);
    comm_init_op->SetAttr("endpoints", endpoints_str);
    comm_init_op->SetAttr("op_role",
                          static_cast<int>(framework::OpRole::kForward));
    comm_init_op->CheckAttrs();
  } else if (config_.use_custom_device()) {
    framework::VarDesc *new_var = block->Var(tmp_var_name);
    new_var->SetType(framework::proto::VarType::RAW);
    new_var->SetPersistable(true);
    framework::OpDesc *gen_bkcl_id_op = block->AppendOp();
    gen_bkcl_id_op->SetType("c_gen_xccl_id");
    gen_bkcl_id_op->SetOutput("Out", {tmp_var_name});
    gen_bkcl_id_op->SetAttr("rank", rank);
    gen_bkcl_id_op->SetAttr("endpoint",
                            config_.dist_config().current_endpoint());
    gen_bkcl_id_op->SetAttr("other_endpoints", peer_endpoints);
    gen_bkcl_id_op->SetAttr("ring_id", ring_id);
    gen_bkcl_id_op->SetAttr("op_role",
                            static_cast<int>(framework::OpRole::kForward));
    gen_bkcl_id_op->CheckAttrs();
    framework::OpDesc *comm_init_op = block->AppendOp();
    comm_init_op->SetType("c_comm_init");
    comm_init_op->SetInput("X", {tmp_var_name});
    comm_init_op->SetAttr("rank", rank);
    comm_init_op->SetAttr("nranks", nranks);
    comm_init_op->SetAttr("ring_id", ring_id);
    comm_init_op->SetAttr("endpoints", endpoints_str);
    comm_init_op->SetAttr("op_role",
                          static_cast<int>(framework::OpRole::kForward));
    comm_init_op->CheckAttrs();
  } else {
    LOG(WARNING) << "DistModelInf doesn't init comm.";
    // TODO(fleet exe dev): comm init for more devices
  }
}

bool AnalysisPredictor::LoadConverterConfig(
    std::map<int64_t, std::vector<int64_t>> *ring_id_to_ranks,
    std::map<int64_t, std::vector<int64_t>> *rank_to_ring_ids) {
  VLOG(3) << "Going to load converter config from: "
          << config_.dist_config().comm_init_config() << "\n";
  std::ifstream fin(config_.dist_config().comm_init_config(), std::ios::in);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin.is_open()),
      true,
      platform::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          config_.dist_config().comm_init_config()));
  std::string line;
  bool ring_to_rank{true};
  // Reading config from file, the config file should like these format
  //  [ring_id -> ranks]
  //  0,0,1,2,3
  //  1,0,1
  //  2,2,3
  //  21,0,1
  //  22,1,2
  //  23,2,3
  //  [rank -> ring_ids]
  //  0,0,1,21
  //  1,0,1,21,22
  //  2,0,2,22,23
  //  3,0,2,23
  while (std::getline(fin, line)) {
    std::vector<std::string> one_line = paddle::string::Split(line, ',');
    if (one_line.size() == 1) {
      // start a new section of the config
      if (line == "[ring_id -> ranks]") {
        ring_to_rank = true;
      } else if (line == "[rank -> ring_ids]") {
        ring_to_rank = false;
      }
    } else {
      // parse key - values pairs in one section
      int64_t key = std::stoll(one_line[0]);
      for (size_t i = 1; i < one_line.size(); ++i) {
        int64_t val = std::stoll(one_line[i]);
        if (ring_to_rank) {
          if (ring_id_to_ranks->find(key) == ring_id_to_ranks->end()) {
            ring_id_to_ranks->insert({key, std::vector<int64_t>()});
          }
          ring_id_to_ranks->at(key).emplace_back(val);
        } else {
          if (rank_to_ring_ids->find(key) == rank_to_ring_ids->end()) {
            rank_to_ring_ids->insert({key, std::vector<int64_t>()});
          }
          rank_to_ring_ids->at(key).emplace_back(val);
        }
        // NOTE: add more configuration sections here
      }
    }
  }
  std::stringstream ss;
  ss << "Loaded the following converter config:\n";
  ss << "ring_id_to_ranks:\n";
  for (auto pair : *ring_id_to_ranks) {
    int64_t key = pair.first;
    ss << "\t" << key << "\t->\t";
    for (auto value : pair.second) {
      ss << value << "\t";
    }
    ss << "\n";
  }
  ss << "rank_to_ring_ids:\n";
  for (auto pair : *rank_to_ring_ids) {
    int64_t key = pair.first;
    ss << "\t" << key << "\t->\t";
    for (auto value : pair.second) {
      ss << value << "\t";
    }
    ss << "\n";
  }
  VLOG(3) << ss.str();
  return true;
}
#endif

void AnalysisPredictor::MkldnnPreSet(const std::vector<PaddleTensor> &inputs) {
#ifdef PADDLE_WITH_DNNL
  std::vector<std::vector<int>> inputs_shape;
  for (const auto &input : inputs) {
    inputs_shape.emplace_back(input.shape);
  }
  MkldnnPreSet(inputs_shape);
#endif
}

void AnalysisPredictor::MkldnnPreSet(
    const std::vector<paddle::Tensor> &inputs) {
#ifdef PADDLE_WITH_DNNL
  std::vector<std::vector<int>> inputs_shape;
  for (const auto &input : inputs) {
    inputs_shape.emplace_back(phi::vectorize<int>(input.dims()));
  }
  MkldnnPreSet(inputs_shape);
#endif
}

void AnalysisPredictor::MkldnnPreSet(
    const std::vector<std::vector<int>> &inputs_shape) {
#ifdef PADDLE_WITH_DNNL
  VLOG(2) << "AnalysisPredictor::ZeroCopyRun get_cur_mkldnn_session_id="
          << phi::OneDNNContext::tls().get_cur_mkldnn_session_id();
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0) {
    VLOG(2) << "In mkldnn cache clear mode.";
    phi::OneDNNContext::tls().set_cur_mkldnn_session_id(
        phi::OneDNNContextThreadLocals::kMKLDNNSessionID_CacheClearing);
    // Set current_input_shape for caching dynamic shape.
    std::stringstream ss;
    for (const auto &input_shape : inputs_shape) {
      for (int item : input_shape) {
        ss << item << "-";
      }
    }
    VLOG(2) << "Set input shape=" << ss.str();
    phi::OneDNNContext::tls().set_cur_input_shape_str(ss.str());
  }
  phi::OneDNNContext::tls().set_cur_input_shape_cache_capacity(
      config_.mkldnn_cache_capacity_);

#endif
}

void AnalysisPredictor::MkldnnPostReset() {
#ifdef PADDLE_WITH_DNNL
  // In cache clearing mode.
  if (config_.mkldnn_cache_capacity_ > 0 &&
      static_cast<phi::OneDNNContext *>(
          (&platform::DeviceContextPool::Instance())->Get(platform::CPUPlace()))
              ->GetCachedObjectsNumber() > 0) {
    if (VLOG_IS_ON(2)) {
      auto shape_blob_size = static_cast<phi::OneDNNContext *>(
                                 (&platform::DeviceContextPool::Instance())
                                     ->Get(platform::CPUPlace()))
                                 ->GetShapeBlobSize();
      CHECK_LE(shape_blob_size,
               static_cast<size_t>(config_.mkldnn_cache_capacity_));
    }
    // We cannot reset to the default cache settings
    // as there maybe CopyToCPU method used and oneDNN
    // primitives are used there so cache would grow
  }
#endif
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPreSet(inputs);
#endif
  VLOG(3) << "Predictor::predict";
  inference::Timer timer;
  timer.tic();
  // set feed variable
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::PreconditionNotMet("The scope should not be nullptr."));
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }

#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
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

  // All the containers in the scope will be hold in inference, but the
  // operators assume that the container will be reset after each batch.
  // Here is a bugfix, collect all the container variables, and reset then to a
  // bool; the next time, the operator will call MutableData and construct a new
  // container again, so that the container will be empty for each batch.
  if (sub_scope_) {
    tensor_array_batch_cleaner_.CollectNoTensorVars(sub_scope_);
  }
  tensor_array_batch_cleaner_.ResetNoTensorVars();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
  return true;
}

bool AnalysisPredictor::Run(const std::vector<paddle::Tensor> &inputs,
                            std::vector<paddle::Tensor> *outputs) {
  inference::DisplayMemoryInfo(place_, "before run");
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPreSet(inputs);
#endif
  VLOG(3) << "predict start";
  // set feed variable
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::PreconditionNotMet("The scope should not be nullptr."));
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }

#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
  }

  // Run the inference program
  // if share variables, we need not create variables
  executor_->Run();

  inference::DisplayMemoryInfo(place_, "after run");

  // get fetch variable
  if (!GetFetch(outputs, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }

  // All the containers in the scope will be hold in inference, but the
  // operators assume that the container will be reset after each batch.
  // Here is a bugfix, collect all the container variables, and reset then to a
  // bool; the next time, the operator will call MutableData and construct a new
  // container again, so that the container will be empty for each batch.
  if (sub_scope_) {
    tensor_array_batch_cleaner_.CollectNoTensorVars(sub_scope_);
  }
  tensor_array_batch_cleaner_.ResetNoTensorVars();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
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
    phi::DenseTensor *input = &feed_tensors_[i];
    if (!PaddleTensorToDenseTensor(inputs[i], input, place_)) {
      return false;
    }
    int idx = -1;
    if (config_.specify_input_name_) {
      auto name = inputs[i].name;
      if (feed_names_.find(name) == feed_names_.end()) {
        LOG(ERROR) << "feed names from program do not have name: [" << name
                   << "] from specified input";
      }
      idx = static_cast<int>(feed_names_[name]);
    } else {
      idx = PADDLE_GET_CONST(int, feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, *input, framework::kFeedOpType, idx);
  }
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<paddle::Tensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  PADDLE_ENFORCE_EQ(inputs.size(),
                    feeds_.size(),
                    platform::errors::InvalidArgument(
                        "wrong feed input size, need %d but get %d.",
                        feeds_.size(),
                        inputs.size()));
  for (const auto &input : inputs) {
    PADDLE_ENFORCE_EQ(input.defined(),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The input Tensor expected to be defined."));
    PADDLE_ENFORCE_EQ(
        input.is_dense_tensor(),
        true,
        paddle::platform::errors::InvalidArgument(
            "The input Tensor expected to be type of dense tensor."));
  }

  if (std::all_of(inputs.cbegin(), inputs.cend(), [&](const paddle::Tensor &t) {
        return !t.name().empty() && feed_names_.count(t.name());
      })) {
    for (const auto &input : inputs) {
      auto &t = framework::GetVariableTensor(*scope, input.name());
      t.ShareDataWith(
          *std::dynamic_pointer_cast<phi::DenseTensor>(input.impl()));
    }
  } else {
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto &t = framework::GetVariableTensor(*scope, idx2feeds_[i]);
      t.ShareDataWith(
          *std::dynamic_pointer_cast<phi::DenseTensor>(inputs[i].impl()));
    }
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const phi::DenseTensor &fetch,
                                    PaddleTensor *output) {
  // set shape.
  auto shape = phi::vectorize(fetch.dims());
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
    int idx = PADDLE_GET_CONST(int, fetches_[i]->GetAttr("col"));
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(idx),
        i,
        platform::errors::InvalidArgument(
            "Fetch op's col attr(%d) should be equal to the index(%d)",
            idx,
            i));
    framework::FetchType &fetch_var =
        framework::GetFetchVariable(*scope, framework::kFetchOpType, idx);
    auto &fetch = PADDLE_GET(phi::DenseTensor, fetch_var);
    auto type = framework::TransToProtoVarType(fetch.dtype());
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
    } else if (type == framework::proto::VarType::FP16) {
      GetFetchOne<float16>(fetch, output);
      output->dtype = PaddleDType::FLOAT16;
    } else if (type == framework::proto::VarType::BF16) {
      GetFetchOne<bfloat16>(fetch, output);
      output->dtype = PaddleDType::BFLOAT16;
    } else {
      LOG(ERROR)
          << "unknown type, only support float32, float16, bfloat16, int64 and "
             "int32 now.";
    }
  }
  return true;
}

bool AnalysisPredictor::GetFetch(std::vector<paddle::Tensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetches_.size());
  for (size_t i = 0; i < fetches_.size(); ++i) {
    auto const &name = idx2fetches_[i];
    auto &t = framework::GetVariableTensor(*scope, name);
    (*outputs)[i] = paddle::Tensor(std::make_shared<phi::DenseTensor>(t), name);
  }
  return true;
}

void AnalysisPredictor::PrepareArgument() {
  VLOG(3) << "AnalysisPredictor::PrepareArgument";
  // Init std::unique_ptr argument_.
  argument_ = std::make_unique<Argument>();
  argument_->SetUseGPU(config_.use_gpu());
  argument_->SetUseCutlass(config_.use_cutlass_);
  argument_->SetUseFcPadding(config_.use_fc_padding());
  argument_->SetGPUDeviceId(config_.gpu_device_id());
  argument_->SetEnableIrOptim(config_.enable_ir_optim_);
  argument_->SetEnableMemoryOptim(config_.enable_memory_optim());
  argument_->SetModelFromMemory(config_.model_from_memory_);
  // Analyze inference_program
  argument_->SetPredictorID(predictor_id_);
  argument_->SetRootPredictorID(root_predictor_id_);
  argument_->SetSaveOptimizedModel(config_.save_optimized_model_);
  argument_->SetOptimCacheDir(config_.opt_cache_dir_);
  if (!config_.model_dir().empty()) {
    argument_->SetModelDir(config_.model_dir());
  } else {
    PADDLE_ENFORCE_EQ(config_.prog_file().empty(),
                      false,
                      platform::errors::PreconditionNotMet(
                          "Either model_dir or prog_file should be set."));

    argument_->SetModelProgramPath(config_.prog_file());
    argument_->SetModelParamsPath(config_.params_file());
  }
  // For JITLayer
  argument_->SetSkipLoadParams(config_.skip_load_params_);

  argument_->SetTensorRtPrecisionMode(static_cast<int>(
      paddle::ConvertPrecision(config_.tensorrt_precision_mode_)));
  argument_->SetTensorRtUseOSS(config_.trt_use_varseqlen_);
  argument_->SetTensorRtWithInterleaved(config_.trt_with_interleaved_);
  argument_->SetTensorRtTransformerPosid(config_.tensorrt_transformer_posid_);
  argument_->SetTensorRtTransformerMaskid(config_.tensorrt_transformer_maskid_);
  argument_->SetMinInputShape(config_.min_input_shape_);
  argument_->SetMaxInputShape(config_.max_input_shape_);
  argument_->SetOptimInputShape(config_.optim_input_shape_);
  argument_->SetTensorRtTunedDynamicShape(
      config_.tuned_tensorrt_dynamic_shape());
  argument_->SetUseTensorRT(false);
  if (config_.use_gpu() && config_.tensorrt_engine_enabled()) {
    LOG(INFO) << "TensorRT subgraph engine is enabled";
    argument_->SetUseTensorRT(true);
    argument_->SetTensorRtWorkspaceSize(config_.tensorrt_workspace_size_);
    argument_->SetTensorRtMaxBatchSize(config_.tensorrt_max_batchsize_);
    argument_->SetTensorRtMinSubgraphSize(config_.tensorrt_min_subgraph_size_);
    argument_->SetTRTMarkOutput(config_.trt_mark_output_);
    argument_->SetTRTOutputTensorNames(config_.trt_output_tensor_names_);
    argument_->SetTensorRtDisabledOPs(config_.trt_disabled_ops_);
    argument_->SetTensorRtUseDLA(config_.trt_use_dla_);
    argument_->SetTensorRtDLACore(config_.trt_dla_core_);
    argument_->SetTensorRtUseStaticEngine(config_.trt_use_static_engine_);
    argument_->SetTensorRtUseCalibMode(config_.trt_use_calib_mode_);
    argument_->SetTensorRtUseCudaGraph(config_.trt_use_cuda_graph_);
    argument_->SetCloseTrtPluginFp16(config_.disable_trt_plugin_fp16_);
    argument_->SetTensorRtShapeRangeInfoPath(config_.shape_range_info_path());
    argument_->SetTensorRtAllowBuildAtRuntime(
        config_.trt_allow_build_at_runtime());
    argument_->SetTensorRtUseInspector(config_.trt_use_inspector_);
    argument_->SetTensorRtInspectorSerialize(config_.trt_inspector_serialize_);
    argument_->SetTensorRtUseExplicitQuantization(
        config_.trt_use_explicit_quantization_);
    argument_->SetTrtEngineMemorySharing(config_.trt_engine_memory_sharing());
    argument_->SetTensorRtOptimizationLevel(config_.trt_optimization_level_);
  }

  if (config_.dlnne_enabled()) {
    LOG(INFO) << "Dlnne subgraph is enabled";
    argument_->SetUseDlnne(true);
    argument_->SetDlnneMinSubgraphSize(config_.dlnne_min_subgraph_size_);
    argument_->SetDlnneMaxBatchSize(config_.dlnne_max_batchsize_);
    argument_->SetDlnneUseStaticBatch(config_.dlnne_use_static_batch_);
    argument_->SetDlnneWeightShareMode(config_.dlnne_weight_share_mode_);
    argument_->SetDlnneDisableNodesByOutputs(
        config_.dlnne_disable_nodes_by_outputs_);
    argument_->SetDlnneInputShapeDict(config_.dlnne_input_shape_dict_);
    argument_->SetDlnneUseCalibMode(config_.dlnne_use_calib_mode_);
    argument_->SetDlnnePrecisionMode(static_cast<int>(
        paddle::ConvertPrecision(config_.dlnne_precision_mode_)));
  }

  argument_->SetUseXpu(config_.use_xpu_);
  if (config_.lite_engine_enabled()) {
    argument_->SetCpuMathLibraryNumThreads(
        config_.cpu_math_library_num_threads());
    argument_->SetLitePrecisionMode(static_cast<int>(
        paddle::ConvertPrecision(config_.lite_precision_mode_)));
    argument_->SetLitePassesFilter(config_.lite_passes_filter_);
    argument_->SetLiteOpsFilter(config_.lite_ops_filter_);
    argument_->SetLiteZeroCopy(config_.lite_zero_copy_);
    argument_->SetXpuLocked(config_.xpu_lite_l3_locked_);
    argument_->SetXpuEnableMultiStream(config_.xpu_lite_enable_multi_stream_);
    argument_->SetUseOpenCL(config_.use_opencl_);
    // NNAdapter related
    argument_->SetUseNNAdapter(config_.NNAdapter().use_nnadapter);
    argument_->SetNNAdapterDeviceNames(
        config_.NNAdapter().nnadapter_device_names);
    argument_->SetNNAdapterContextProperties(
        config_.NNAdapter().nnadapter_context_properties);
    argument_->SetNNAdapterModelCacheDir(
        config_.NNAdapter().nnadapter_model_cache_dir);
    argument_->SetNNAdapterSubgraphPartitionConfigBuffer(
        config_.NNAdapter().nnadapter_subgraph_partition_config_buffer);
    argument_->SetNNAdapterSubgraphPartitionConfigPath(
        config_.NNAdapter().nnadapter_subgraph_partition_config_path);
    std::vector<std::string> buffer_keys;
    std::vector<std::vector<char>> buffer_vals;
    for (auto it : config_.NNAdapter().nnadapter_model_cache_buffers) {
      buffer_keys.emplace_back(it.first);
      buffer_vals.emplace_back(it.second);
    }
    argument_->SetNNAdapterModelCacheToken(buffer_keys);
    argument_->SetNNAdapterModelCacheBuffer(buffer_vals);
    LOG(INFO) << "Lite subgraph engine is enabled";
  }

#ifdef PADDLE_WITH_IPU
  argument_->SetUseIpu(config_.use_ipu_);
  argument_->SetIpuDeviceNum(config_.ipu_device_num());
  argument_->SetIpuMicroBatchSize(config_.ipu_micro_batch_size_);
  argument_->SetIpuEnablePipelining(config_.ipu_enable_pipelining_);
  argument_->SetIpuBatchesPerStep(config_.ipu_batches_per_step_);
  argument_->SetIpuEnableFp16(config_.ipu_enable_fp16_);
  argument_->SetIpuReplicaNum(config_.ipu_replica_num_);
  argument_->SetIpuAvailableMemoryProportion(
      config_.ipu_available_memory_proportion_);
  argument_->SetIpuEnableHalfPartial(config_.ipu_enable_half_partial_);
  argument_->SetIpuEnableModelRuntimeExecutor(
      config_.ipu_enable_model_runtime_executor_);
  argument_->SetIpuCustomOpsInfo(config_.ipu_custom_ops_info_);
  argument_->SetIpuCustomPatterns(config_.ipu_custom_patterns_);
#endif

  if (config_.use_mkldnn_) {
    LOG(INFO) << "MKLDNN is enabled";
    argument_->SetMKLDNNEnabledOpTypes(config_.mkldnn_enabled_op_types_);
  }

  if (config_.use_cinn_compiler_) {
    argument_->SetUseCinnCompiler(config_.use_cinn_compiler_);
  }

#ifdef PADDLE_WITH_DNNL
  if (config_.mkldnn_quantizer_enabled()) {
    LOG(INFO) << "Quantization is enabled";
    argument_->SetQuantizeEnabledOpTypes(
        config_.mkldnn_quantizer_config()->enabled_op_types());
    argument_->SetQuantizeExcludedOpIds(
        config_.mkldnn_quantizer_config()->excluded_op_ids());
  }
  if (config_.use_mkldnn_bfloat16_) {
    LOG(INFO) << "Bfloat16 is enabled";
    argument_->SetBfloat16EnabledOpTypes(config_.bfloat16_enabled_op_types_);
  }

  if (config_.use_mkldnn_int8_) {
    LOG(INFO) << "Int8 is enabled";
    argument_->SetQuantizeEnabledOpTypes(config_.quantize_enabled_op_types_);
    argument_->SetQuantizeExcludedOpIds(config_.quantize_excluded_op_ids_);
    argument_->SetQuantVarScales({});
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  argument_->SetUseCustomDevice(config_.use_custom_device());
  if (config_.use_custom_device()) {
    LOG(INFO) << "CustomDevice is enabled";
    argument_->SetCustomDeviceType(config_.custom_device_type());
    argument_->SetCustomDeviceId(config_.custom_device_id());
  }
#endif

  argument_->SetUseXpu(config_.use_xpu_);
  argument_->SetXpuDeviceId(config_.xpu_config_.device_id);
  argument_->SetXpuL3Size(config_.xpu_config_.l3_size);
  argument_->SetXpuL3Ptr(config_.xpu_config_.l3_ptr);
  argument_->SetXpuL3AutotuneSize(config_.xpu_config_.l3_autotune_size);
  argument_->SetXpuContextGmSize(config_.xpu_config_.context_gm_size);
  argument_->SetXpuContext(config_.xpu_config_.context);
  argument_->SetXpuStream(config_.xpu_config_.stream);
  argument_->SetXpuConvAutotuneLevel(config_.xpu_config_.conv_autotune_level);
  argument_->SetXpuConvAutotuneFile(config_.xpu_config_.conv_autotune_file);
  argument_->SetXpuConvAutotuneFileWriteback(
      config_.xpu_config_.conv_autotune_file_writeback);
  argument_->SetXpuFcAutotuneLevel(config_.xpu_config_.fc_autotune_level);
  argument_->SetXpuFcAutotuneFile(config_.xpu_config_.fc_autotune_file);
  argument_->SetXpuFcAutotuneFileWriteback(
      config_.xpu_config_.fc_autotune_file_writeback);
  argument_->SetXpuGemmComputePrecision(
      config_.xpu_config_.gemm_compute_precision);
  argument_->SetXpuTransformerSoftmaxOptimizeLevel(
      config_.xpu_config_.transformer_softmax_optimize_level);
  argument_->SetXpuTransformerEncoderAdaptiveSeqlen(
      config_.xpu_config_.transformer_encoder_adaptive_seqlen);
  argument_->SetXpuQuantPostStaticGeluOutThreshold(
      config_.xpu_config_.quant_post_static_gelu_out_threshold);
  argument_->SetXpuQuantPostDynamicActivationMethod(
      config_.xpu_config_.quant_post_dynamic_activation_method);
  argument_->SetXpuQuantPostDynamicWeightPrecision(
      config_.xpu_config_.quant_post_dynamic_weight_precision);
  argument_->SetXpuQuantPostDynamicOpTypes(
      config_.xpu_config_.quant_post_dynamic_op_types);
  argument_->SetXpuLiteL3Locked(config_.xpu_lite_l3_locked_);
  argument_->SetXpuLiteEnableMultiStream(config_.xpu_lite_enable_multi_stream_);

  auto *pass_builder = config_.pass_builder();
  // TODO(inference): Need to reconstruct the pass_builder, pass should be
  // processed in a single
  if (model_precision_ != phi::DataType::FLOAT32) {
    LOG(INFO) << "Model is mixed precision type with " << model_precision_
              << ", we will use a new PassStrategy. Note that only GPU/XPU "
                 "backend is supported for now.";
    if (!config_.use_cinn_compiler_) {
      const auto &deleted_passes = pass_builder->GetAllDeletedPasses();
      if (config_.tensorrt_engine_enabled()) {
        pass_builder->ClearPasses();
        for (const auto &pass : kTrtLowerPrecisionPasses) {
          if (deleted_passes.count(pass)) continue;
          pass_builder->AppendPass(pass);
        }
      } else if (config_.use_gpu()) {
        pass_builder->ClearPasses();
        for (const auto &pass : kGpuLowerPrecisionPasses) {
          if (deleted_passes.count(pass)) continue;
          pass_builder->AppendPass(pass);
        }
      } else if (config_.use_xpu()) {
        // All passes support fp16. Not reset pass_builder.
      } else if (config_.use_custom_device()) {
        // All passes support fp16. Not reset pass_builder.
      } else {
        pass_builder->ClearPasses();
      }
    }
  }

  if (!config_.ir_optim()) {
    argument_->SetEnableIrOptim(false);
    if (config_.enable_gpu_mixed_ &&
        model_precision_ == phi::DataType::FLOAT32) {
      argument_->SetEnableIrOptim(true);
      pass_builder->ClearPasses();
      pass_builder->AppendPass("auto_mixed_precision_pass");
      LOG(INFO) << "This model run in GPU mixed precision mode with no ir "
                   "optimization.";
    } else {
      LOG(INFO)
          << "Ir optimization is turned off, no ir pass will be executed.";
    }
  } else {
    if (config_.ir_debug_) {
      pass_builder->TurnOnDebug();
    }
    if (config_.enable_gpu_mixed_) {
      LOG(INFO) << "This model run in GPU mixed precision mode.";
    }
  }

  argument_->SetEnableCustomDeviceMixed(config_.enable_custom_device_mixed());
  if (config_.enable_custom_device_mixed_) {
    argument_->SetEnableIrOptim(true);
    pass_builder->ClearPasses();
    pass_builder->AppendPass("auto_mixed_precision_pass");
    LOG(INFO) << "This model run in Custom Device mixed precision mode.";
  }

  argument_->SetDisableLogs(config_.glog_info_disabled());
  argument_->SetIrAnalysisPasses(pass_builder->AllPasses());
  argument_->SetAnalysisPasses(pass_builder->AnalysisPasses());
  argument_->SetScopeNotOwned(scope_.get());

  // mixed precison.
  argument_->SetModelPrecision(static_cast<int>(model_precision_));
  argument_->SetMixedBlackList(config_.mixed_black_list_);
  argument_->SetMixedWhiteList(config_.mixed_white_list_);
  argument_->SetEnableGPUMixed(config_.enable_gpu_mixed_);
  argument_->SetMixedPrecisionMode(static_cast<int>(
      paddle::ConvertPrecision(config_.mixed_precision_mode_)));
  argument_->SetEnableLowPrecisionIO(config_.enable_low_precision_io_);
}

// NOTE All the members in AnalysisConfig should be copied to Argument.
void AnalysisPredictor::OptimizeInferenceProgram() {
  PrepareArgument();
#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif
  Analyzer().Run(argument_.get());
  PADDLE_ENFORCE_EQ(
      argument_->scope_valid(),
      true,
      platform::errors::InvalidArgument("The argument scope should be valid."));
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((argument_.get()), ir_analyzed_program);
  inference_program_.reset(
      new framework::ProgramDesc(argument_->ir_analyzed_program()),
      [](framework::ProgramDesc *prog) {
// Note, please do NOT use any member variables, because member variables may
// have been destructed in multiple threads.
#ifdef PADDLE_WITH_TENSORRT
        auto &block = prog->Block(0);
        for (auto &op_desc : block.AllOps()) {
          if (op_desc->Type() == "tensorrt_engine") {
            std::string engine_key =
                PADDLE_GET_CONST(std::string, op_desc->GetAttr("engine_key"));
            int engine_predictor_id =
                PADDLE_GET_CONST(int, op_desc->GetAttr("predictor_id"));
            std::string engine_name =
                engine_key + std::to_string(engine_predictor_id);
            if (paddle::inference::Singleton<
                    inference::tensorrt::TRTEngineManager>::Global()
                    .Has(engine_name)) {
              paddle::inference::Singleton<
                  inference::tensorrt::TRTEngineManager>::Global()
                  .DeleteKey(engine_name);
            }
          }
        }
#endif
        delete prog;
      });
  // The config and argument take a lot of storage,
  // when the predictor settings are complete, we release these stores.
  config_.PartiallyRelease();
#if defined(PADDLE_WITH_TESTING)
  fusion_statis_ = *argument_->fusion_statis_ptr();
#endif

#if defined(_WIN32)
  argument_->PartiallyRelease();
#else
  if (config_.mkldnn_enabled() ||
      (config_.tensorrt_engine_enabled() &&
       config_.tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8)) {
    argument_->PartiallyRelease();
  } else {
    argument_.reset(nullptr);
  }
#endif
  LOG(INFO) << "======= optimize end =======";
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
    const AnalysisConfig &config) {
  PADDLE_ENFORCE_EQ(
      config.is_valid(),
      true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  // Register custom operators compiled by the user.
  // This function can only be executed once per process.
  static std::once_flag custom_operators_registered;
  std::call_once(custom_operators_registered,
                 []() { inference::RegisterAllCustomOperator(); });

  auto SetGflags = [](const AnalysisConfig &config) {
    auto SetGflag = [](const char *name, const char *value) {
      bool success = paddle::flags::SetFlagValue(name, value);
      PADDLE_ENFORCE_EQ(
          success,
          true,
          platform::errors::InvalidArgument(
              "Fail to set gflag: %s, please make sure the gflag exists.",
              name));
      VLOG(3) << "set gflag: --" << name << "=" << value;
    };
    // TODO(NHZlX): Should add the link to the doc of
    // paddle_infer::CreatePredictor<paddle_infer::Config>
    if (config.glog_info_disabled()) {
      FLAGS_logtostderr = true;
      FLAGS_minloglevel = 2;  // GLOG_ERROR
    }

    if (config.use_gpu()) {
      static std::once_flag gflags_initialized;
      static bool process_level_allocator_enabled;

      std::call_once(gflags_initialized, [&]() {
        PADDLE_ENFORCE_GE(
            config.memory_pool_init_size_mb(),
            0.f,
            platform::errors::InvalidArgument(
                "The size of memory pool should be greater than 0."));
        PADDLE_ENFORCE_GE(config.gpu_device_id(),
                          0,
                          platform::errors::InvalidArgument(
                              "Invalid device id (%d). The device id should be "
                              "greater than 0.",
                              config.gpu_device_id()));

        float fraction_of_gpu_memory = config.fraction_of_gpu_memory_for_pool();
        if (fraction_of_gpu_memory > 0.95f) {
          LOG(ERROR)
              << "Allocate too much memory for the GPU memory pool, assigned "
              << config.memory_pool_init_size_mb() << " MB";
          LOG(ERROR) << "Try to shink the value by setting "
                        "AnalysisConfig::EnableUseGpu(...)";
        }
        if (fraction_of_gpu_memory >= 0.0f || fraction_of_gpu_memory <= 0.95f) {
          std::string value = std::to_string(fraction_of_gpu_memory);
          SetGflag("fraction_of_gpu_memory_to_use", value.data());
        }

        // TODO(Shixiaowei02): Add a mandatory scheme to use the thread local
        // allocator when multi-stream is enabled.
        if (config.thread_local_stream_enabled()) {
          SetGflag("allocator_strategy", "thread_local");
          process_level_allocator_enabled = false;
        } else {
          process_level_allocator_enabled = true;
        }

        // for inference, the following default values are better.
        if (std::getenv("FLAGS_conv_workspace_size_limit") == nullptr) {
          SetGflag("conv_workspace_size_limit", "32");
        }
        if (std::getenv("FLAGS_initial_cpu_memory_in_mb") == nullptr) {
          SetGflag("initial_cpu_memory_in_mb", "0");
        }
      });

      if (config.thread_local_stream_enabled() &&
          process_level_allocator_enabled) {
        PADDLE_THROW(platform::errors::Fatal(
            "When binding threads and streams, the use of "
            "process-level allocators will result in undefined result "
            "errors due to memory asynchronous operations."
            "The thread and stream binding configuration of all "
            "predictors should be the same in a single process."));
      }
    }
  };
  SetGflags(config);

  VLOG(3) << "create AnalysisPredictor";

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<AnalysisPredictor *>(predictor.get());

#ifdef PADDLE_WITH_TENSORRT
  paddle::framework::ir::patterns::KeyCounter::Instance().CleanCounter();
#endif

  if (!predictor_p->Init(nullptr)) {
    return nullptr;
  }

  if (config.mkldnn_quantizer_enabled() && !predictor_p->MkldnnQuantize()) {
    return nullptr;
  }

  return predictor;
}

bool AnalysisPredictor::MkldnnQuantize() {
#if PADDLE_WITH_DNNL
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
  PADDLE_ENFORCE_NOT_NULL(sub_scope_,
                          platform::errors::InvalidArgument(
                              "The sub_scope should not be nullptr."));
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == framework::kFeedOpType) {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx2feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == framework::kFetchOpType) {
      int idx = PADDLE_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      idx2fetches_[idx] = op->Input("X")[0];
    }
  }
}

void AnalysisPredictor::CreateFeedFetchVar(framework::Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument("The scope should not be nullptr."));
  auto *var = scope->Var(framework::kFeedOpType);
  var->GetMutable<framework::FeedList>();
  var = scope->Var(framework::kFetchOpType);
  var->GetMutable<framework::FetchList>();
}

std::vector<std::string> AnalysisPredictor::GetInputNames() {
  std::vector<std::string> input_names;
  for (auto &item : idx2feeds_) {
    input_names.push_back(item.second);
  }
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
AnalysisPredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  std::vector<std::string> names = GetInputNames();
  for (std::string name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet("Input %s does not exist.", name));
    input_shapes[name] = var->GetShape();
  }
  return input_shapes;
}

std::map<std::string, paddle_infer::DataType>
AnalysisPredictor::GetInputTypes() {
  std::map<std::string, paddle_infer::DataType> input_type;
  std::vector<std::string> names = GetInputNames();
  for (const auto &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet(
            "Input %s does not exist inference_program_.", name));
    auto dtype = var->GetDataType();
    if (dtype == paddle::framework::proto::VarType::FP32) {
      input_type[name] = paddle_infer::DataType::FLOAT32;
    } else if (dtype == paddle::framework::proto::VarType::FP16) {
      input_type[name] = paddle_infer::DataType::FLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::BF16) {
      input_type[name] = paddle_infer::DataType::BFLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::INT64) {
      input_type[name] = paddle_infer::DataType::INT64;
    } else if (dtype == paddle::framework::proto::VarType::INT32) {
      input_type[name] = paddle_infer::DataType::INT32;
    } else if (dtype == paddle::framework::proto::VarType::UINT8) {
      input_type[name] = paddle_infer::DataType::UINT8;
    } else if (dtype == paddle::framework::proto::VarType::INT8) {
      input_type[name] = paddle_infer::DataType::INT8;
    } else if (dtype == paddle::framework::proto::VarType::FP64) {
      input_type[name] = paddle_infer::DataType::FLOAT64;
    } else if (dtype == paddle::framework::proto::VarType::BOOL) {
      input_type[name] = paddle_infer::DataType::BOOL;
    } else {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data type `%s` when get input dtype ", dtype));
    }
  }
  return input_type;
}

std::vector<std::string> AnalysisPredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  for (auto &item : idx2fetches_) {
    output_names.push_back(item.second);
  }
  return output_names;
}

std::map<std::string, std::vector<int64_t>>
AnalysisPredictor::GetOutputTensorShape() {
  std::map<std::string, std::vector<int64_t>> output_shapes;
  std::vector<std::string> names = GetOutputNames();
  for (std::string name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(var,
                            platform::errors::PreconditionNotMet(
                                "Output %s does not exist.", name));
    output_shapes[name] = var->GetShape();
  }
  return output_shapes;
}

std::map<std::string, paddle_infer::DataType>
AnalysisPredictor::GetOutputTypes() {
  std::map<std::string, paddle_infer::DataType> output_type;
  std::vector<std::string> names = GetOutputNames();
  for (const auto &name : names) {
    auto *var = inference_program_->Block(0).FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet(
            "Output %s does not exist inference_program_.", name));
    auto dtype = var->GetDataType();
    if (dtype == paddle::framework::proto::VarType::FP32) {
      output_type[name] = paddle_infer::DataType::FLOAT32;
    } else if (dtype == paddle::framework::proto::VarType::FP16) {
      output_type[name] = paddle_infer::DataType::FLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::BF16) {
      output_type[name] = paddle_infer::DataType::BFLOAT16;
    } else if (dtype == paddle::framework::proto::VarType::INT64) {
      output_type[name] = paddle_infer::DataType::INT64;
    } else if (dtype == paddle::framework::proto::VarType::INT32) {
      output_type[name] = paddle_infer::DataType::INT32;
    } else if (dtype == paddle::framework::proto::VarType::UINT8) {
      output_type[name] = paddle_infer::DataType::UINT8;
    } else if (dtype == paddle::framework::proto::VarType::INT8) {
      output_type[name] = paddle_infer::DataType::INT8;
    } else {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported data type `%s` when get output dtype ", dtype));
    }
  }
  return output_type;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  framework::Scope *scope;
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  if (config_.dist_config().use_dist_model()) {
    scope = scope_.get();
  } else {
    scope = executor_->GetScope();
  }
#else
  scope = executor_->GetScope();
#endif
  PADDLE_ENFORCE_NOT_NULL(
      scope->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the executor.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(
      static_cast<void *>(scope), this->GetDeviceContexts()));
  res->input_or_output_ = true;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {  // NOLINT
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_ipu_place(place_)) {
    // Currently, IPUPlace's tensor copy between cpu and ipu has been set in
    // IpuBackend.
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = place_;
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else if (platform::is_custom_place(place_)) {
    auto custom_place = place_;
    res->SetPlace(PaddlePlace::kCUSTOM,
                  custom_place.GetDeviceId(),
                  custom_place.GetDeviceType());
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  framework::Scope *scope;
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  if (config_.dist_config().use_dist_model()) {
    scope = scope_.get();
  } else {
    scope = executor_->GetScope();
  }
#else
  scope = executor_->GetScope();
#endif
  PADDLE_ENFORCE_NOT_NULL(
      scope->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The variable named %s is not found in the scope of the executor.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(
      static_cast<void *>(scope), this->GetDeviceContexts()));
  res->input_or_output_ = false;
  res->SetName(name);
  if (platform::is_cpu_place(place_)) {  // NOLINT
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_ipu_place(place_)) {
    // Currently, IPUPlace's tensor copy between cpu and ipu has been set in
    // IpuBackend.
    res->SetPlace(PaddlePlace::kCPU);
  } else if (platform::is_xpu_place(place_)) {
    if (config_.lite_engine_enabled()) {
      // Currently, Paddle-Lite's XPU user interface only supports the transfer
      // of host data pointers. If it is currently used as a subgraph, execution
      // efficiency will be sacrificed, so it is temporarily set to cpu place.
      // And, the current lite engine of xpu must execute all parts of the
      // model.
      res->SetPlace(PaddlePlace::kCPU);
    } else {
      auto xpu_place = place_;
      res->SetPlace(PaddlePlace::kXPU, xpu_place.GetDeviceId());
    }
  } else if (platform::is_custom_place(place_)) {
    auto custom_place = place_;
    res->SetPlace(PaddlePlace::kCUSTOM,
                  custom_place.GetDeviceId(),
                  custom_place.GetDeviceType());
  } else {
    auto gpu_place = place_;
    res->SetPlace(PaddlePlace::kGPU, gpu_place.GetDeviceId());
  }
  return res;
}

bool AnalysisPredictor::ZeroCopyRun() {
  inference::DisplayMemoryInfo(place_, "before run");
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  if (config_.dist_config().use_dist_model()) {
    VLOG(3) << "ZeroCopyRun will use the fleet executor.";
    inference::Timer timer;
    timer.tic();
    fleet_exe_->Run(config_.dist_config().carrier_id());
    VLOG(3) << "Fleet executor inf runs once use: "
            << std::to_string(timer.toc()) << "ms";
    return true;
  }
#endif
  if (private_context_) {
    paddle::platform::DeviceContextPool::SetDeviceContexts(&device_contexts_);
  }
  paddle::platform::SetNumThreads(config_.cpu_math_library_num_threads());
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) {
    std::vector<std::vector<int>> shape_vector;
    auto names = GetInputNames();
    for (auto &name : names) {
      auto in_tensor = GetInputTensor(name);
      shape_vector.emplace_back(in_tensor->shape());
    }
    MkldnnPreSet(shape_vector);
  }
#endif

#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled()) {
    inference::tensorrt::TensorRTEngine::predictor_id_per_thread =
        predictor_id_;
    VLOG(3) << "thread_local var predictor_id in TensorRTEngine is set to: "
            << inference::tensorrt::TensorRTEngine::predictor_id_per_thread;
  }
#endif

  if (config_.shape_range_info_collected()) {
    HookCollectShapeRangeInfo();
  }
#ifdef PADDLE_WITH_XPU
  InferXPUContext *infer_xpu_ctx = nullptr;
  if (config_.use_xpu_ && !config_.use_lite_) {
    PADDLE_ENFORCE(
        private_context_,
        paddle::platform::errors::Fatal(
            "Must use private context if run predictor on xpu place."));
    auto *dev_ctxs = reinterpret_cast<const std::map<
        phi::Place,
        std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        this->GetDeviceContexts());
    infer_xpu_ctx =
        static_cast<InferXPUContext *>(dev_ctxs->at(place_).get().get());
    auto *x_context = static_cast<xpu::Context *>(config_.xpu_config_.context);
    if (x_context != nullptr) {
      infer_xpu_ctx->SetXContext(x_context);
    }
    infer_xpu_ctx->SetStream(predictor_stream_);
    infer_xpu_ctx->SetL3Info(config_.xpu_config_.l3_size,
                             config_.xpu_config_.l3_ptr,
                             config_.xpu_config_.l3_autotune_size,
                             place_);
  }
#endif

  executor_->Run();
  inference::DisplayMemoryInfo(place_, "after run");

#ifdef PADDLE_WITH_XPU
  if (config_.use_xpu_ && !config_.use_lite_ && infer_xpu_ctx != nullptr) {
    infer_xpu_ctx->L3CacheAutotune();
  }
#endif

  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(sub_scope_);
  tensor_array_batch_cleaner_.ResetTensorArray();

  // recover the cpu_math_library_num_threads to 1, in order to avoid thread
  // conflict when integrating it into deployment service.
  paddle::platform::SetNumThreads(1);
  if (private_context_) {
    paddle::platform::DeviceContextPool::SetDeviceContexts(nullptr);
  }
#ifdef PADDLE_WITH_DNNL
  if (config_.use_mkldnn_) MkldnnPostReset();
#endif
#if defined(PADDLE_WITH_MKLML)
  // Frees unused memory allocated by the Intel® MKL Memory Allocator to
  // avoid memory leak. See:
  // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-free-buffers
  phi::dynload::MKL_Free_Buffers();
#endif
  return true;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
bool AnalysisPredictor::ExpRunWithExternalStream(const gpuStream_t stream) {
  if (!private_context_) {
    PADDLE_THROW(platform::errors::Fatal(
        "Please use config.SetExecStream to init gpu resources, and then we "
        "will bind gpu resources to execution stream."));
  }

  if (stream != predictor_stream_) {
#ifdef PADDLE_WITH_HIP
    hipStreamSynchronize(static_cast<gpuStream_t>(predictor_stream_));
#else
    cudaStreamSynchronize(static_cast<gpuStream_t>(predictor_stream_));
#endif
    ResourceManager::Instance().GpuResourceSwitchStream(predictor_stream_,
                                                        stream);
    predictor_stream_ = stream;

    auto *dev_ctxs = const_cast<
        std::map<phi::Place,
                 std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        reinterpret_cast<const std::map<
            phi::Place,
            std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
            this->GetDeviceContexts()));

    dev_ctxs->erase(place_);
    dev_ctxs->emplace(
        place_, std::async(std::launch::deferred, [=] {
          auto *gpu_resource =
              ResourceManager::Instance().GetGPUResource(predictor_stream_);
          auto *gpu_context = new InferGPUContext(place_);
          UpdatePrivateDeviceContext(gpu_context, gpu_resource, place_);
          return std::unique_ptr<phi::DeviceContext>(gpu_context);
        }));
    auto &pool = paddle::experimental::DeviceContextPool::Instance();
    pool.SyncDeviceContext(place_);
  }

  return ZeroCopyRun();
}
#endif

void AnalysisPredictor::HookCollectShapeRangeInfo() {
  auto hook = [&](const std::string &op_type,
                  const std::string &input_name,
                  const paddle::Tensor &var) -> void {
    paddle::platform::DeviceContextPool &pool =
        paddle::platform::DeviceContextPool::Instance();
    if (config_.use_gpu()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto *dev_ctx = pool.Get(place_);
      auto stream = static_cast<phi::GPUContext *>(dev_ctx)->stream();
#ifdef PADDLE_WITH_HIP
      hipStreamSynchronize(stream);
#else
      cudaStreamSynchronize(stream);
#endif
#endif
    }

    auto *new_var = sub_scope_->GetVar(input_name);
    if (!new_var) return;
    if (!new_var->IsType<phi::DenseTensor>()) {
      return;
    }
    auto tensor = new_var->Get<phi::DenseTensor>();
    if (!tensor.initialized()) return;
    framework::DDim dim = tensor.dims();
    std::vector<int32_t> shape(dim.size());
    for (int i = 0; i < static_cast<int>(shape.size()); ++i)
      shape[i] = static_cast<int32_t>(dim[i]);
    if (!shape.empty()) {
      shape_info_[input_name].emplace_back(shape);
    } else if (tensor.numel() > 0) {
      // This must be a zero dimension tensor.
      PADDLE_ENFORCE_EQ(tensor.numel(),
                        1UL,
                        platform::errors::PreconditionNotMet(
                            "This tensor must have one element, but got %ld.",
                            tensor.numel()));
      std::vector<int32_t> zero_shape(1, 1);
      shape_info_[input_name].emplace_back(zero_shape);
    }

    // We need collect value range for shape tensor for Paddle-TRT's use.
    // To be noticed, this method to identify all shape tensors is based on
    // assumption that all shape tensors in the model have numbers <= 8.
    // This is a simple method to identify all shape tensors with some
    // mistakes, but it doesn't matter.
    auto is_shape_tensor = tensor.numel() <= 8 && tensor.numel() >= 1;
    if ((tensor.dtype() == phi::DataType::INT32 ||
         tensor.dtype() == phi::DataType::INT64) &&
        is_shape_tensor) {
      std::vector<int> int32_host(tensor.numel());

      if (platform::is_cpu_place(tensor.place())) {
        auto &int32_tensor = tensor;
        if (tensor.dtype() == phi::DataType::INT64) {
          auto *cpu_ctx = pool.Get(platform::CPUPlace());
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::CPUContext &>(*cpu_ctx),
              tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(platform::CPUPlace(),
                             int32_host.data(),
                             platform::CPUPlace(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int));
      } else if (platform::is_gpu_place(tensor.place())) {
#if defined(PADDLE_WITH_CUDA)
        auto *dev_ctx = pool.Get(tensor.place());
        auto &int32_tensor = tensor;
        if (tensor.dtype() == phi::DataType::INT64) {
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::GPUContext &>(*dev_ctx),
              tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(platform::CPUPlace(),
                             int32_host.data(),
                             int32_tensor.place(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int),
                             nullptr);
#endif
      }
      shape_tensor_value_[input_name].emplace_back(int32_host);
    }
  };
  RegisterInputHook(hook);
}

bool AnalysisPredictor::ExpRunWithRuntimeConfig(void *config) {
#ifdef PADDLE_WITH_XPU
  auto xpu_runtime_config =
      reinterpret_cast<paddle_infer::experimental::XpuRuntimeConfig *>(config);

  config_.xpu_config_.context = xpu_runtime_config->context;
  auto *stream = xpu_runtime_config->stream;
  if (stream != nullptr && stream != predictor_stream_) {
    paddle::platform::XPUStreamSync(
        static_cast<paddle::xpuStream>(predictor_stream_));
    predictor_stream_ = stream;
  }

  auto l3_size = xpu_runtime_config->l3_size;
  auto l3_autotune_size = xpu_runtime_config->l3_autotune_size;
  PADDLE_ENFORCE_LE(
      l3_autotune_size,
      l3_size,
      phi::errors::InvalidArgument(
          "l3_autotune_size(%zu) should be less than or equal to l3_size(%zu).",
          l3_autotune_size,
          l3_size));
  config_.xpu_config_.l3_size = l3_size;
  config_.xpu_config_.l3_ptr = xpu_runtime_config->l3_ptr;
  config_.xpu_config_.l3_autotune_size = l3_autotune_size;

  return ZeroCopyRun();
#endif
  return false;
}

void AnalysisPredictor::StatisticShapeRangeInfo() {
  std::map<std::string, std::vector<int32_t>> min_shapes;
  std::map<std::string, std::vector<int32_t>> max_shapes;
  std::map<std::string, std::vector<int32_t>> opt_shapes;
  std::map<std::string, std::vector<int32_t>> min_values;
  std::map<std::string, std::vector<int32_t>> max_values;
  std::map<std::string, std::vector<int32_t>> opt_values;

  auto extract_min_max_opt =
      [](std::map<std::string, std::vector<int32_t>> &min_data,
         decltype(min_data) max_data,
         decltype(min_data) opt_data,
         decltype(shape_info_) shape_data) {
        for (auto it : shape_data) {
          auto name = it.first;
          auto shapes = it.second;

          std::vector<int32_t> min_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> max_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> opt_shape(shapes[0].begin(), shapes[0].end());

          auto ShapeMaxFreq =
              [](const std::map<int32_t, int32_t> &m) -> int32_t {
            std::vector<std::pair<int32_t, int32_t>> counter;
            for (auto &it : m) counter.emplace_back(it);
            std::sort(counter.begin(),
                      counter.end(),
                      [](std::pair<int32_t, int32_t> &a,
                         std::pair<int32_t, int32_t> &b) {
                        return a.second > b.second;
                      });
            return counter[0].first;
          };

          for (size_t d = 0; d < shapes[0].size(); ++d) {
            std::map<int32_t, int32_t> counter;
            for (auto &shape : shapes) {
              counter[shape[d]] += 1;
              if (shape[d] < min_shape[d]) min_shape[d] = shape[d];
              if (shape[d] > max_shape[d]) max_shape[d] = shape[d];
            }
            opt_shape[d] = ShapeMaxFreq(counter);
          }

          min_data[name] = min_shape;
          max_data[name] = max_shape;
          opt_data[name] = opt_shape;
        }
      };
  extract_min_max_opt(min_shapes, max_shapes, opt_shapes, shape_info_);
  extract_min_max_opt(min_values, max_values, opt_values, shape_tensor_value_);

  inference::SerializeShapeRangeInfo(config_.shape_range_info_path(),
                                     min_shapes,
                                     max_shapes,
                                     opt_shapes,
                                     min_values,
                                     max_values,
                                     opt_values);
}

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir().empty()) {
    filename = config_.model_dir() + "/__model__";
  } else if (!config_.prog_file().empty()) {
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
        "not valid model path '%s' or program path '%s'.",
        config_.model_dir(),
        config_.params_file());
    return false;
  }

  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  if (!config_.model_from_memory()) {
    std::string pb_content;
    // Read binary
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin.is_open()),
        true,
        platform::errors::NotFound(
            "Cannot open file %s, please confirm whether the file is normal.",
            filename));
    fin.seekg(0, std::ios::end);
    pb_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(pb_content.at(0)), pb_content.size());  // NOLINT
    fin.close();

    proto.ParseFromString(pb_content);
  } else {
    proto.ParseFromString(config_.prog_file());
  }
  inference_program_ = std::make_unique<framework::ProgramDesc>(proto);
  return true;
}

bool AnalysisPredictor::LoadParameters() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));

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

uint64_t AnalysisPredictor::TryShrinkMemory() {
  ClearIntermediateTensor();
  return paddle::memory::Release(place_);
}

void AnalysisPredictor::ClearIntermediateTensor() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The inference program should be loaded first."));
  const auto &global_block = inference_program_->MutableBlock(0);
  for (auto *var : global_block->AllVars()) {
    if (!IsPersistable(var)) {
      const std::string name = var->Name();
      auto *variable = executor_->GetScope()->FindVar(name);
      if (variable != nullptr && variable->IsType<phi::DenseTensor>() &&
          name != framework::kFeedOpType && name != framework::kFetchOpType) {
        VLOG(3) << "Clear Intermediate Tensor: " << name;
        auto *t = variable->GetMutable<phi::DenseTensor>();
        t->clear();
      }
    }
  }
}

#ifdef PADDLE_WITH_TENSORRT
using inference::Singleton;
bool AnalysisPredictor::SaveTrtCalibToDisk() {
  PADDLE_ENFORCE_EQ(config_.tensorrt_engine_enabled(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "This func can be invoked only in trt mode"));
  auto &block = inference_program_->Block(0);
  for (auto &op_desc : block.AllOps()) {
    if (op_desc->Type() == "tensorrt_engine") {
      std::string engine_name = PADDLE_GET_CONST(
          std::string, op_desc->GetAttr("calibration_engine_key"));
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
          argument_->Has("model_dir") ? argument_->model_dir()
                                      : inference::analysis::GetDirRoot(
                                            argument_->model_program_path());

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

AnalysisPredictor::~AnalysisPredictor() {  // NOLINT
#ifdef PADDLE_WITH_TENSORRT
  if (config_.tensorrt_engine_enabled() &&
      config_.tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8 &&
      Singleton<TRTCalibratorEngineManager>::Global().Has()) {
    SaveTrtCalibToDisk();
  }
#endif
  if (config_.with_profile_) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
  if (sub_scope_) {
    if (framework::global_transfer_scope_key().find(sub_scope_) !=
        framework::global_transfer_scope_key().end()) {
      auto scope_key_set = framework::global_transfer_scope_key()[sub_scope_];
      for (auto item : scope_key_set) {
        framework::global_transfer_data_cache().erase(item);
      }
      framework::global_transfer_scope_key().erase(sub_scope_);
    }
    scope_->DeleteScope(sub_scope_);
  }

#if PADDLE_WITH_DNNL
  if (mkldnn_quantizer_) {
    delete mkldnn_quantizer_;
    mkldnn_quantizer_ = nullptr;
  }
#endif

  if (config_.shape_range_info_collected()) {
    StatisticShapeRangeInfo();
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (predictor_stream_ != nullptr) {
    ResourceManager::Instance().DestroyGPUResource(predictor_stream_);
  }
#endif

  if (place_.GetType() != phi::AllocationType::UNDEFINED) {
    memory::Release(place_);
  }
  device_contexts_.clear();

#ifdef PADDLE_WITH_TENSORRT
  if (config_.trt_engine_memory_sharing()) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .ReleaseContextMemory(predictor_id_);
  }
#endif
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone(void *stream) {
  VLOG(3) << "AnalysisPredictor::Clone";
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new AnalysisPredictor(config_);
  x->status_is_cloned_ = true;
  x->root_predictor_id_ = this->root_predictor_id_;
  x->config_.apply_optim_ = false;
  if (config_.use_external_stream_ && stream == nullptr) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "config has been configured to use external stream, but the Clone "
        "function has not received a valid stream parameter."));
  } else if (!config_.use_external_stream_ && stream != nullptr) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "config has not been configured to use external stream, but the Clone "
        "function has received a stream parameter."));
  }
  x->predictor_stream_ = stream;
  x->Init(scope_, inference_program_);
#ifdef PADDLE_WITH_TENSORRT
  x->executor_->ResetTrtOps(++AnalysisPredictor::clone_num_);
#endif
#ifdef PADDLE_WITH_LITE
#ifdef LITE_SUBGRAPH_WITH_XPU
  x->executor_->CloneLiteEnigne(++AnalysisPredictor::clone_num_,
                                config_.xpu_config_.stream);
#else
  x->executor_->CloneLiteEnigne(++AnalysisPredictor::clone_num_, nullptr);
#endif
#endif
  return std::unique_ptr<PaddlePredictor>(x);
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

void AnalysisPredictor::RegisterInputHook(const InputTensorHookFunc &hookfunc) {
  std::call_once(register_input_hook_flag_, [this] {
    executor_->RegisterInputHook(
        [this](framework::OperatorBase *op, framework::Scope *scope) {
          for (auto &input : op->Inputs()) {
            for (auto &var_name : input.second) {
              auto *var = scope->FindVar(var_name);
              if (!var || !var->IsType<phi::DenseTensor>()) continue;
              auto dense_tensor = var->Get<phi::DenseTensor>();
              if (!dense_tensor.initialized()) continue;
              auto tensor = paddle::Tensor(
                  std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
              for (auto &hookfunc : this->input_hookfuncs_) {
                hookfunc(op->Type(), var_name, tensor);
              }
            }
          }
        });
  });
  input_hookfuncs_.push_back(hookfunc);
}

void AnalysisPredictor::RegisterOutputHook(
    const OutputTensorHookFunc &hookfunc) {
  std::call_once(register_output_hook_flag_, [this] {
    executor_->RegisterOutputHook(
        [this](framework::OperatorBase *op, framework::Scope *scope) {
          for (auto &output : op->Outputs()) {
            for (auto &var_name : output.second) {
              auto *var = scope->FindVar(var_name);
              if (!var || !var->IsType<phi::DenseTensor>()) continue;
              auto dense_tensor = var->Get<phi::DenseTensor>();
              if (!dense_tensor.initialized()) continue;
              auto tensor = paddle::Tensor(
                  std::make_shared<phi::DenseTensor>(dense_tensor), var_name);
              for (auto &hookfunc : this->output_hookfuncs_) {
                hookfunc(op->Type(), var_name, tensor);
              }
            }
          }
        });
  });
  output_hookfuncs_.push_back(hookfunc);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<AnalysisConfig>(
    const AnalysisConfig &config) {
  LOG(WARNING) << "Deprecated. Please use CreatePredictor instead.";
  return CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
      config);
}

}  // namespace paddle

#ifdef PADDLE_WITH_TENSORRT
USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(elementwise_sub_weight);
USE_TRT_CONVERTER(elementwise_mul_weight);
USE_TRT_CONVERTER(elementwise_div_weight);
USE_TRT_CONVERTER(elementwise_min_weight);
USE_TRT_CONVERTER(elementwise_max_weight);
USE_TRT_CONVERTER(elementwise_pow_weight);
USE_TRT_CONVERTER(elementwise_mod_weight);
USE_TRT_CONVERTER(elementwise_floordiv_weight);
USE_TRT_CONVERTER(elementwise_add_tensor);
USE_TRT_CONVERTER(elementwise_sub_tensor);
USE_TRT_CONVERTER(elementwise_div_tensor);
USE_TRT_CONVERTER(elementwise_mul_tensor);
USE_TRT_CONVERTER(elementwise_max_tensor);
USE_TRT_CONVERTER(elementwise_min_tensor);
USE_TRT_CONVERTER(elementwise_pow_tensor);
USE_TRT_CONVERTER(elementwise_floordiv_tensor);
USE_TRT_CONVERTER(elementwise_mod_tensor);
USE_TRT_CONVERTER(less_than);
USE_TRT_CONVERTER(greater_than);
USE_TRT_CONVERTER(logical_or);
USE_TRT_CONVERTER(logical_xor);
USE_TRT_CONVERTER(logical_and);
USE_TRT_CONVERTER(less_equal);
USE_TRT_CONVERTER(greater_equal);
USE_TRT_CONVERTER(transpose);
USE_TRT_CONVERTER(transpose2);
USE_TRT_CONVERTER(flatten);
USE_TRT_CONVERTER(flatten_contiguous_range);
USE_TRT_CONVERTER(matrix_multiply);
USE_TRT_CONVERTER(bmm);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(sigmoid);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
USE_TRT_CONVERTER(batch_norm);
USE_TRT_CONVERTER(concat);
USE_TRT_CONVERTER(dropout);
USE_TRT_CONVERTER(pad);
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(pad3d);
USE_TRT_CONVERTER(einsum)
#endif
USE_TRT_CONVERTER(hard_sigmoid);
USE_TRT_CONVERTER(hard_swish);
USE_TRT_CONVERTER(split);
USE_TRT_CONVERTER(fill_any_like);
USE_TRT_CONVERTER(prelu);
USE_TRT_CONVERTER(conv2d_transpose);
USE_TRT_CONVERTER(leaky_relu);
USE_TRT_CONVERTER(shuffle_channel);
USE_TRT_CONVERTER(where);
USE_TRT_CONVERTER(bitwise_not);
USE_TRT_CONVERTER(one_hot);
USE_TRT_CONVERTER(one_hot_v2);
USE_TRT_CONVERTER(swish);
USE_TRT_CONVERTER(silu);
USE_TRT_CONVERTER(group_norm);
USE_TRT_CONVERTER(instance_norm);
USE_TRT_CONVERTER(layer_norm);
USE_TRT_CONVERTER(gelu);
USE_TRT_CONVERTER(multihead_matmul);
USE_TRT_CONVERTER(multihead_matmul_roformer);
USE_TRT_CONVERTER(skip_layernorm);
USE_TRT_CONVERTER(slice);
USE_TRT_CONVERTER(scale);
USE_TRT_CONVERTER(stack);
USE_TRT_CONVERTER(clip);
USE_TRT_CONVERTER(gather);
USE_TRT_CONVERTER(anchor_generator);
USE_TRT_CONVERTER(yolo_box);
USE_TRT_CONVERTER(yolo_box_head);
USE_TRT_CONVERTER(arg_max);
USE_TRT_CONVERTER(arg_min);
USE_TRT_CONVERTER(roi_align);
USE_TRT_CONVERTER(affine_channel);
USE_TRT_CONVERTER(multiclass_nms);
USE_TRT_CONVERTER(multiclass_nms3);
USE_TRT_CONVERTER(nearest_interp);
USE_TRT_CONVERTER(nearest_interp_v2);
USE_TRT_CONVERTER(bilinear_interp_v2);
USE_TRT_CONVERTER(reshape);
USE_TRT_CONVERTER(reshape2);
USE_TRT_CONVERTER(gather_nd);
USE_TRT_CONVERTER(reduce_mean);
USE_TRT_CONVERTER(reduce_max);
USE_TRT_CONVERTER(reduce_min);
USE_TRT_CONVERTER(reduce_sum);
USE_TRT_CONVERTER(reduce_prod);
USE_TRT_CONVERTER(reduce_any);
USE_TRT_CONVERTER(reduce_all);
USE_TRT_CONVERTER(tile);
USE_TRT_CONVERTER(conv3d);
USE_TRT_CONVERTER(conv3d_transpose);
USE_TRT_CONVERTER(mish);
USE_TRT_CONVERTER(deformable_conv);
USE_TRT_CONVERTER(pool3d)
USE_TRT_CONVERTER(square);
// unary op
USE_TRT_CONVERTER(exp);
USE_TRT_CONVERTER(log);
USE_TRT_CONVERTER(sqrt);
USE_TRT_CONVERTER(reciprocal);
USE_TRT_CONVERTER(abs);
USE_TRT_CONVERTER(sin);
USE_TRT_CONVERTER(cos);
USE_TRT_CONVERTER(tan);
USE_TRT_CONVERTER(sinh);
USE_TRT_CONVERTER(cosh);
USE_TRT_CONVERTER(tanh);
USE_TRT_CONVERTER(asin);
USE_TRT_CONVERTER(acos);
USE_TRT_CONVERTER(atan);
USE_TRT_CONVERTER(asinh);
USE_TRT_CONVERTER(acosh);
USE_TRT_CONVERTER(atanh);
USE_TRT_CONVERTER(ceil);
USE_TRT_CONVERTER(floor);
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(round);
USE_TRT_CONVERTER(sign);
#endif
USE_TRT_CONVERTER(rsqrt);
USE_TRT_CONVERTER(fused_preln_embedding_eltwise_layernorm)
USE_TRT_CONVERTER(prompt_tuning_emb_eltwise_layernorm);
USE_TRT_CONVERTER(fused_embedding_eltwise_layernorm);
USE_TRT_CONVERTER(preln_skip_layernorm)
USE_TRT_CONVERTER(fused_bias_dropout_residual_layer_norm)
USE_TRT_CONVERTER(c_allreduce_sum)
USE_TRT_CONVERTER(roll)
USE_TRT_CONVERTER(strided_slice)
USE_TRT_CONVERTER(rnn)
USE_TRT_CONVERTER(fill_constant_batch_size_like)
USE_TRT_CONVERTER(transformer_input_convert)
USE_TRT_CONVERTER(cast)
USE_TRT_CONVERTER(recover_padding)
USE_TRT_CONVERTER(remove_padding)
USE_TRT_CONVERTER(equal);
USE_TRT_CONVERTER(not_equal);
USE_TRT_CONVERTER(top_k)
USE_TRT_CONVERTER(top_k_v2)
USE_TRT_CONVERTER(range)
USE_TRT_CONVERTER(squeeze2)
USE_TRT_CONVERTER(unsqueeze2)
USE_TRT_CONVERTER(sum)
USE_TRT_CONVERTER(shape)
USE_TRT_CONVERTER(fill_constant)
USE_TRT_CONVERTER(fused_token_prune)
USE_TRT_CONVERTER(celu)
USE_TRT_CONVERTER(layernorm_shift_partition)
USE_TRT_CONVERTER(reverse_roll)
USE_TRT_CONVERTER(preln_layernorm_shift_partition)
USE_TRT_CONVERTER(merge_layernorm)
USE_TRT_CONVERTER(trans_layernorm)
USE_TRT_CONVERTER(skip_merge_layernorm)
USE_TRT_CONVERTER(generic_plugin_creater)
USE_TRT_CONVERTER(custom_plugin_creater)
USE_TRT_CONVERTER(fuse_eleadd_transpose)
USE_TRT_CONVERTER(tanh_shrink)
USE_TRT_CONVERTER(logsigmoid)
USE_TRT_CONVERTER(lookup_table)
USE_TRT_CONVERTER(lookup_table_v2)
USE_TRT_CONVERTER(expand_v2)
USE_TRT_CONVERTER(expand_as_v2)
USE_TRT_CONVERTER(take_along_axis)
USE_TRT_CONVERTER(skip_groupnorm_act)
USE_TRT_CONVERTER(preln_groupnorm_act)
USE_TRT_CONVERTER(cumsum)
USE_TRT_CONVERTER(assign)
USE_TRT_CONVERTER(unbind)
USE_TRT_CONVERTER(flip)
#if IS_TRT_VERSION_GE(8522)
USE_TRT_CONVERTER(flash_multihead_matmul)
USE_TRT_CONVERTER(cross_multihead_matmul)
USE_TRT_CONVERTER(qk_multihead_matmul)
#endif
#if IS_TRT_VERSION_GE(8510)
USE_TRT_CONVERTER(grid_sampler)
#endif
#if IS_TRT_VERSION_GE(8200)
USE_TRT_CONVERTER(set_value)
USE_TRT_CONVERTER(index_select);
USE_TRT_CONVERTER(temporal_shift)
#endif
#if PADDLE_WITH_CUSPARSELT && IS_TRT_VERSION_GE(8000)
USE_TRT_CONVERTER(sparse_fc)
USE_TRT_CONVERTER(sparse_multihead_matmul)
#endif
#if IS_TRT_VERSION_GE(8000)
USE_TRT_CONVERTER(quantize_linear)
USE_TRT_CONVERTER(dequantize_linear)
#endif
#endif

namespace paddle_infer {

Predictor::Predictor(const Config &config) {
  const_cast<Config *>(&config)->SwitchUseFeedFetchOps(false);
  // The second parameter indicates that the discard log is not printed
  if (config.use_onnxruntime()) {
#ifdef PADDLE_WITH_ONNXRUNTIME
    if (config.use_gpu()) {
      LOG(WARNING) << "The current ONNXRuntime backend doesn't support GPU,"
                      "and it falls back to use Paddle Inference.";
    } else if (!paddle::CheckConvertToONNX(config)) {
      LOG(WARNING)
          << "Paddle2ONNX do't support convert the Model, fall back to using "
             "Paddle Inference.";
    } else {
      predictor_ =
          paddle::CreatePaddlePredictor<Config,
                                        paddle::PaddleEngineKind::kONNXRuntime>(
              config);
      return;
    }
#else
    LOG(WARNING)
        << "The onnxruntime backend isn't enabled,"
           " and please re-compile Paddle with WITH_ONNXRUNTIME option,"
           "fall back to using Paddle Inference.";
#endif
  }
  predictor_ =
      paddle::CreatePaddlePredictor<Config,
                                    paddle::PaddleEngineKind::kAnalysis>(
          config);
}

std::vector<std::string> Predictor::GetInputNames() {
  return predictor_->GetInputNames();
}

std::map<std::string, std::vector<int64_t>> Predictor::GetInputTensorShape() {
  return predictor_->GetInputTensorShape();
}

std::map<std::string, DataType> Predictor::GetInputTypes() {
  return predictor_->GetInputTypes();
}

std::unique_ptr<Tensor> Predictor::GetInputHandle(const std::string &name) {
  return predictor_->GetInputTensor(name);
}

std::vector<std::string> Predictor::GetOutputNames() {
  return predictor_->GetOutputNames();
}

std::unique_ptr<Tensor> Predictor::GetOutputHandle(const std::string &name) {
  return predictor_->GetOutputTensor(name);
}

std::map<std::string, std::vector<int64_t>> Predictor::GetOutputTensorShape() {
  return predictor_->GetOutputTensorShape();
}

std::map<std::string, DataType> Predictor::GetOutputTypes() {
  return predictor_->GetOutputTypes();
}

bool Predictor::Run() { return predictor_->ZeroCopyRun(); }

bool Predictor::Run(const std::vector<paddle::Tensor> &inputs,
                    std::vector<paddle::Tensor> *outputs) {
  return predictor_->Run(inputs, outputs);
}

std::unique_ptr<Predictor> Predictor::Clone(void *stream) {
  auto analysis_pred = predictor_->Clone(stream);
  std::unique_ptr<Predictor> pred(new Predictor(std::move(analysis_pred)));
  return pred;
}

void Predictor::ClearIntermediateTensor() {
  predictor_->ClearIntermediateTensor();
}

uint64_t Predictor::TryShrinkMemory() { return predictor_->TryShrinkMemory(); }

void Predictor::RegisterOutputHook(const OutputTensorHookFunc &hookfunc) {
  predictor_->RegisterOutputHook(hookfunc);
}
void Predictor::RegisterInputHook(const OutputTensorHookFunc &hookfunc) {
  predictor_->RegisterInputHook(hookfunc);
}

void *Predictor::GetExecStream() const { return predictor_->GetExecStream(); }

int GetNumBytesOfDataType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

std::string GetVersion() { return paddle::get_version(); }

std::tuple<int, int, int> GetTrtCompileVersion() {
#ifdef PADDLE_WITH_TENSORRT
  return paddle::inference::tensorrt::GetTrtCompileVersion();
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

std::tuple<int, int, int> GetTrtRuntimeVersion() {
#ifdef PADDLE_WITH_TENSORRT
  return paddle::inference::tensorrt::GetTrtRuntimeVersion();
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

void UpdateDllFlag(const char *name, const char *value) {
  paddle::UpdateDllFlag(name, value);
}

void ConvertToMixedPrecision(const std::string &model_file,
                             const std::string &params_file,
                             const std::string &mixed_model_file,
                             const std::string &mixed_params_file,
                             PrecisionType mixed_precision,
                             paddle_infer::PlaceType backend,
                             bool keep_io_types,
                             std::unordered_set<std::string> black_list,
                             std::unordered_set<std::string> white_list) {
  auto phi_backend = paddle::ConvertBackend(backend);
  auto phi_precision = paddle::ConvertPrecision(mixed_precision);
  paddle::inference::analysis::ConvertToMixedPrecision(model_file,
                                                       params_file,
                                                       mixed_model_file,
                                                       mixed_params_file,
                                                       phi_precision,
                                                       phi_backend,
                                                       keep_io_types,
                                                       black_list,
                                                       white_list);
}

}  // namespace paddle_infer

namespace paddle_infer {
std::shared_ptr<Predictor> CreatePredictor(const Config &config) {  // NOLINT
  std::shared_ptr<Predictor> predictor(new Predictor(config));
  return predictor;
}

namespace services {
PredictorPool::PredictorPool(const Config &config, size_t size) {
  PADDLE_ENFORCE_GE(
      size,
      1UL,
      paddle::platform::errors::InvalidArgument(
          "The predictor pool size should be greater than 1, but it's (%d)",
          size));
  Config copy_config(config);
  main_pred_ = std::make_unique<Predictor>(config);
  for (size_t i = 0; i < size - 1; i++) {
    if (config.tensorrt_engine_enabled()) {
      Config config_tmp(copy_config);
      preds_.emplace_back(new Predictor(config_tmp));
    } else {
      preds_.emplace_back(main_pred_->Clone());
    }
  }
}

Predictor *PredictorPool::Retrive(size_t idx) {
  PADDLE_ENFORCE_LT(
      idx,
      preds_.size() + 1,
      paddle::platform::errors::InvalidArgument(
          "There are (%d) predictors in the pool, but the idx is (%d)",
          idx,
          preds_.size() + 1));
  if (idx == 0) {
    return main_pred_.get();
  }
  return preds_[idx - 1].get();
}
}  // namespace services

namespace experimental {

// Note: Can only be used under thread_local semantics.
bool InternalUtils::RunWithExternalStream(paddle_infer::Predictor *p,
                                          cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithExternalStream(stream);
#endif
  return false;
}
bool InternalUtils::RunWithExternalStream(paddle_infer::Predictor *p,
                                          hipStream_t stream) {
#ifdef PADDLE_WITH_HIP
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithExternalStream(stream);
#endif
  return false;
}

bool InternalUtils::RunWithRuntimeConfig(paddle_infer::Predictor *p,
                                         void *config) {
  auto pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  return pred->ExpRunWithRuntimeConfig(config);
}

void InternalUtils::UpdateConfigInterleaved(paddle_infer::Config *c,
                                            bool with_interleaved) {
#ifdef PADDLE_WITH_CUDA
  c->trt_with_interleaved_ = with_interleaved;
#endif
}

void InternalUtils::SetTransformerPosid(
    paddle_infer::Config *c, const std::string &tensorrt_transformer_posid) {
#ifdef PADDLE_WITH_CUDA
  c->tensorrt_transformer_posid_ = tensorrt_transformer_posid;
#endif
}

void InternalUtils::SetTransformerMaskid(
    paddle_infer::Config *c, const std::string &tensorrt_transformer_maskid) {
#ifdef PADDLE_WITH_CUDA
  c->tensorrt_transformer_maskid_ = tensorrt_transformer_maskid;
#endif
}

void InternalUtils::SyncStream(paddle_infer::Predictor *p) {
#ifdef PADDLE_WITH_CUDA
  auto *pred = dynamic_cast<paddle::AnalysisPredictor *>(p->predictor_.get());
  paddle::platform::DeviceContextPool &pool =
      paddle::platform::DeviceContextPool::Instance();
  auto *dev_ctx = reinterpret_cast<phi::GPUContext *>(pool.Get(pred->place_));
  cudaStreamSynchronize(dev_ctx->stream());
#endif
}
void InternalUtils::SyncStream(cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  cudaStreamSynchronize(stream);
#endif
}

}  // namespace experimental
}  // namespace paddle_infer
