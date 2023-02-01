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
#include "paddle/fluid/imperative/tracer.h"

#include <map>
#include <set>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/fluid/platform/denormal.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(use_mkldnn);
DECLARE_string(tracer_mkldnn_ops_on);
DECLARE_string(tracer_mkldnn_ops_off);

namespace paddle {
namespace imperative {

thread_local bool Tracer::enable_program_desc_tracing_ = false;

thread_local bool Tracer::has_grad_ = true;

thread_local bool Tracer::use_layout_autotune_ = false;

thread_local AmpLevel Tracer::amp_level_ = AmpLevel::O0;

thread_local phi::DataType Tracer::amp_dtype_ = phi::DataType::FLOAT32;

static std::shared_ptr<Tracer> g_current_tracer(nullptr);

const std::shared_ptr<Tracer>& GetCurrentTracer() { return g_current_tracer; }

void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer) {
  g_current_tracer = tracer;
  VLOG(6) << "Set current tracer: " << g_current_tracer;
}

void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& pair : outs) {
    for (const auto& var : pair.second) {
      // NOTE(zhiqiu): this happends when None output are passed from python
      // side. For example, fake_quantize_dequantize_moving_average_abs_max may
      // pass None OutAccum in eval mode.
      // It can be refined by generate several different pybind interface for
      // one operator with different function signature.
      if (var == nullptr) {
        VLOG(4) << pair.first << " is NULL";
        continue;
      }
      VLOG(6) << "Set output: " << var->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      var->InnerSetOverridedStopGradient(generate_grad);
    }
  }
}

void IncreaseVarbaseReferenceCountUntilCopyComplete(
    const std::shared_ptr<imperative::VarBase>& var,
    const platform::Place& place) {
  // Note(zhiqiu): Follow the logic of TensorCopy to determine the place that we
  // need to add callback, see tensor_utils.cc:245
  auto place_ = platform::is_gpu_place(place) ? place : var->Place();

  auto tracer = imperative::GetCurrentTracer();
  auto gc = tracer->MutableGarbageCollectorIfNotExists(place_);

  // Note(zhiqiu): This is an empty callback, the only way is to "reference"
  // var, so it will not be destructed until the kernels launched at current
  // stream of given place is finished.
  auto callback = [var, place_]() {
    VLOG(4) << "Run callback of var:" << var->Name() << " at place " << place_;
  };

  gc->DirectClearCallback(callback);
}

paddle::framework::GarbageCollector* Tracer::MutableGarbageCollectorIfNotExists(
    const platform::Place& place) {
  // if not exists, create a new GarbageCollector at given place
  if (gcs_.count(place) == 0) {
    std::unique_ptr<framework::GarbageCollector> gc;
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc.reset(new framework::DefaultStreamGarbageCollector(place, 0));

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use CUDA device since it's not compiled with CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (platform::is_cuda_pinned_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc.reset(new framework::CUDAPinnedGarbageCollector(place, 0));

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use CUDAPinned device since it's not compiled with "
          "CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (platform::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
      gc.reset(new framework::XPUGarbageCollector(place, 0));
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use XPU device since it's not compiled with XPU,"
          "Please recompile or reinstall Paddle with XPU support."));
#endif
    } else if (platform::is_cpu_place(place)) {
      gc.reset(new framework::CPUGarbageCollector(place, 0));
      VLOG(10) << "Created GarbageCollector at " << place;
    } else if (platform::is_npu_place(place)) {
#if defined(PADDLE_WITH_ASCEND_CL)
      // TODO(zhiqiu): fix bugs and enable NPUDefaultStreamGarbageCollector.
      gc.reset(new framework::NPUUnsafeFastGarbageCollector(place, 0));
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use NPU device since it's not compiled with NPU,"
          "Please recompile or reinstall Paddle with NPU support."));
#endif
    } else if (platform::is_ipu_place(place)) {
#if defined(PADDLE_WITH_IPU)
      gc.reset(new framework::IPUGarbageCollector(place, 0));
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use IPU device since it's not compiled with IPU,"
          "Please recompile or reinstall Paddle with IPU support."));
#endif
    } else if (platform::is_mlu_place(place)) {
#if defined(PADDLE_WITH_MLU)
      gc.reset(new framework::MLUDefaultStreamGarbageCollector(place, 0));
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use MLU device since it's not compiled with MLU,"
          "Please recompile or reinstall Paddle with MLU support."));
#endif
    } else if (platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      if (framework::IsFastEagerDeletionModeEnabled()) {
        gc.reset(
            new framework::CustomDeviceUnsafeFastGarbageCollector(place, 0));
        VLOG(10) << "Created UnsafeFastGarbageCollector at " << place;
      } else {
        gc.reset(new framework::CustomDefaultStreamGarbageCollector(place, 0));
        VLOG(10) << "Created GarbageCollector at " << place;
      }
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use CustomDevice since it's not compiled with "
          "CustomDevice,"
          "Please recompile or reinstall Paddle with CustomDevice "
          "support."));
#endif
    } else {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "Unsupported place for garbage collection"));
    }
    gcs_.emplace(place, std::move(gc));
  }

  return gcs_.at(place).get();
}

template <typename VarType>
void Tracer::TraceOp(const std::string& type,
                     const NameVarMap<VarType>& ins,
                     const NameVarMap<VarType>& outs,
                     framework::AttributeMap attrs,
                     const platform::Place& place,
                     bool trace_backward,
                     const std::map<std::string, std::string>& inplace_map,
                     paddle::framework::AttributeMap* passed_default_attrs_,
                     bool use_default_attr_map) {
  TraceOpImpl<VarType>(type,
                       ins,
                       outs,
                       attrs,
                       place,
                       trace_backward,
                       inplace_map,
                       passed_default_attrs_,
                       use_default_attr_map);
}

template <typename VarType>
void Tracer::TraceOpImpl(const std::string& type,
                         const NameVarMap<VarType>& ins,
                         const NameVarMap<VarType>& outs,
                         framework::AttributeMap& attrs,
                         const platform::Place& place,
                         bool trace_backward,
                         const std::map<std::string, std::string>& inplace_map,
                         paddle::framework::AttributeMap* passed_default_attrs_,
                         bool use_default_attr_map) {
  platform::RecordEvent op_type_record_event(
      type, platform::TracerEventType::Operator, 1);
  platform::ScopedFlushDenormal flush;
  VLOG(4) << "Trace Op: " << type;
  if (FLAGS_use_mkldnn) {
    // if both lists are empty all ops are enabled (default for
    // FLAGS_use_mkldnn=1)
    // if ops_on list is not empty only ops from that list are enabled
    if (!FLAGS_tracer_mkldnn_ops_on.empty()) {
      auto is_on = FLAGS_tracer_mkldnn_ops_on.find(type) != std::string::npos;
      attrs["use_mkldnn"] = is_on;
    } else {
      // if ops_on list is empty all ops are enabled except types from off_list
      auto is_off = FLAGS_tracer_mkldnn_ops_off.find(type) != std::string::npos;
      attrs["use_mkldnn"] = !is_off;
    }
  }
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
  }
  const auto& extra_attr_checkers =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(type);
  for (const auto& checker : extra_attr_checkers) {
    checker(&attrs, true);
  }

  static paddle::framework::AttributeMap empty_attrs_map = {};
  const paddle::framework::AttributeMap& default_attrs =
      attr_checker == nullptr ? empty_attrs_map
                              : attr_checker->GetDefaultAttrMap();

  std::unique_ptr<NameVarMap<VarType>> ins_amp = nullptr;
  if (amp_level_ == AmpLevel::O1) {
    if (amp_dtype_ == phi::DataType::FLOAT16) {
      VLOG(5) << "Float16 Auto Mixed Precision O1 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          AutoCastInputs<VarType>(type, ins));
    } else if (amp_dtype_ == phi::DataType::BFLOAT16) {
      VLOG(5) << "BFloat16 Auto Mixed Precision O1 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          AutoCastBF16Inputs<VarType>(type, ins));
    }
  } else if (amp_level_ == AmpLevel::O2) {
    if (amp_dtype_ == phi::DataType::FLOAT16) {
      VLOG(5) << "Float16 Auto Mixed Precision O2 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          CastPureFp16Inputs<VarType>(type, ins));
    } else if (amp_dtype_ == phi::DataType::BFLOAT16) {
      VLOG(5) << "BFloat16 Auto Mixed Precision O2 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          CastPureBf16Inputs<VarType>(type, ins));
    }
  }

  if (platform::is_gpu_place(place)) {
    const auto& new_tmp = ins_amp == nullptr ? ins : *ins_amp;
    const auto& tracer = imperative::GetCurrentTracer();
    ins_amp = std::make_unique<NameVarMap<VarType>>(
        imperative::AutoTuneLayout<VarType>(
            type, new_tmp, outs, &attrs, tracer));
  }

  const auto& new_ins = ins_amp == nullptr ? ins : *ins_amp;

  try {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::SetDeviceId(place.device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
      platform::SetXPUDeviceId(place.device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    } else if (platform::is_npu_place(place)) {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::SetNPUDeviceId(place.device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with NPU if use NPUPlace."));
#endif
    } else if (platform::is_mlu_place(place)) {
#ifdef PADDLE_WITH_MLU
      platform::SetMLUDeviceId(place.device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with MLU if use MLUPlace."));
#endif
    } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      phi::DeviceManager::SetDevice(place);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with CustomDevice if use "
          "CustomPlace."));
#endif
    }
    if (!use_default_attr_map) {
      PADDLE_ENFORCE_NOT_NULL(passed_default_attrs_,
                              paddle::platform::errors::PermissionDenied(
                                  "Detected default_attrs = nullptr."));
      VLOG(6) << "Use passed in default attrs";
      OpBase::Run(*op, new_ins, outs, attrs, (*passed_default_attrs_), place);
    } else {
      VLOG(6) << "Use Checker's default attrs";
      if (passed_default_attrs_) {
        // TODO(jiabin): Update this without copy
        *passed_default_attrs_ = default_attrs;
      }
      OpBase::Run(*op, new_ins, outs, attrs, default_attrs, place);
    }
  } catch (platform::EnforceNotMet& exception) {
    framework::AppendErrorOpHint(type, &exception);
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(
        platform::errors::Fatal("Operator %s raises an %s exception.\n"
                                "The exception content is\n:%s.",
                                type,
                                platform::demangle(typeid(ex).name()),
                                ex.what()));
  } catch (...) {
    // NOTE: this branch represents a very serious bug with
    // low probability of occurrence, and we can't get its
    // exception content here.
    PADDLE_THROW(platform::errors::Fatal(
        "Operator %s raises an unknown exception.", type));
  }

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, new_ins, outs, attrs);
  }

  {
    platform::RecordEvent node_creation_record_event(
        "grad_node_creation", platform::TracerEventType::OperatorInner, 1);

    if (ComputeRequiredGrad(new_ins, outs, trace_backward)) {
      PADDLE_ENFORCE_EQ(
          passed_default_attrs_,
          nullptr,
          paddle::platform::errors::PermissionDenied(
              "We expect passed_default_attrs_ is nullptr while "
              "use_default_attr_map is true, however we got not null "
              "passed_default_attrs_. Please check your usage of trace_op. "));
      CreateGradOpNode(
          *op, new_ins, outs, attrs, default_attrs, place, inplace_map);
    } else {
      VLOG(3) << "No Grad to track for Op: " << type;
    }
    VLOG(6) << "Finish Trace Op: " << type;
  }
}

template void Tracer::TraceOp<VarBase>(
    const std::string& type,
    const NameVarMap<VarBase>& ins,
    const NameVarMap<VarBase>& outs,
    framework::AttributeMap attrs,
    const platform::Place& place,
    bool trace_backward,
    const std::map<std::string, std::string>& inplace_map,
    paddle::framework::AttributeMap* default_attrs,
    bool use_default_attr_map);

template void Tracer::TraceOp<egr::EagerVariable>(
    const std::string& type,
    const NameVarMap<egr::EagerVariable>& ins,
    const NameVarMap<egr::EagerVariable>& outs,
    framework::AttributeMap attrs,
    const platform::Place& place,
    bool trace_backward,
    const std::map<std::string, std::string>& inplace_map_,
    paddle::framework::AttributeMap* default_attrs,
    bool use_default_attr_map);

void Tracer::TraceOp(const std::string& type,
                     const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs,
                     framework::AttributeMap attrs,
                     const std::map<std::string, std::string>& inplace_map) {
  TraceOp<VarBase>(type,
                   ins,
                   outs,
                   std::move(attrs),
                   expected_place_,
                   has_grad_,
                   inplace_map);
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap& attrs,
                     const paddle::platform::Place& place,
                     paddle::framework::AttributeMap* default_attrs,
                     bool use_default_attr_map,
                     const std::map<std::string, std::string>& inplace_map) {
  VLOG(6) << "Running On Eager TraceOp with use_default_attr_map: "
          << use_default_attr_map;
  TraceOpImpl<egr::EagerVariable>(type,
                                  ins,
                                  outs,
                                  attrs,
                                  place,
                                  false,
                                  inplace_map,
                                  default_attrs,
                                  use_default_attr_map);
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap attrs) {
  VLOG(6) << "Running On Eager TraceOp(4 agrs): ";
  TraceOpImpl<egr::EagerVariable>(
      type, ins, outs, attrs, expected_place_, false, {}, nullptr, true);
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap& attrs,
                     const std::map<std::string, std::string>& inplace_map) {
  VLOG(6) << "Running On Eager TraceOp(less): ";
  TraceOpImpl<egr::EagerVariable>(type,
                                  ins,
                                  outs,
                                  attrs,
                                  expected_place_,
                                  false,
                                  inplace_map,
                                  nullptr,
                                  true);
}

void Tracer::SetExpectedPlace(platform::Place place) {
  expected_place_ = place;
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap& outs,
                                 bool trace_backward) {
  if (!trace_backward) return false;

  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverridedStopGradient());
        return true;
      }
    }
  }
  return false;
}

bool Tracer::ComputeRequiredGrad(const NameTensorMap& ins,
                                 const NameTensorMap& outs,
                                 bool trace_backward) {
  return false;
}

phi::KernelSignature Tracer::GetExpectedKernelSignature(
    const std::string& type,
    const NameTensorMap& ins,
    const NameTensorMap& outs,
    framework::AttributeMap attrs) const {
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  framework::RuntimeContext ctx({}, {});
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(phi::CPUPlace());
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
  }
  static paddle::framework::AttributeMap empty_attrs_map = {};
  const paddle::framework::AttributeMap& default_attrs =
      attr_checker == nullptr ? empty_attrs_map
                              : attr_checker->GetDefaultAttrMap();
  auto dygraph_exe_ctx =
      imperative::DygraphExecutionContext<egr::EagerVariable>(
          *op,
          framework::Scope(),
          *dev_ctx,
          ctx,
          ins,
          outs,
          attrs,
          default_attrs);
  auto* opbase_with_kernel =
      dynamic_cast<framework::OperatorWithKernel*>(op.get());
  PADDLE_ENFORCE_NE(opbase_with_kernel,
                    nullptr,
                    platform::errors::InvalidArgument(
                        "This op type:`%s` is not a OperatorWithKernel, only "
                        "OperatorWithKernel can get KernelSignature",
                        type));
  return phi::KernelSignature(
      std::move(opbase_with_kernel->GetExpectedPhiKernelArgs(dygraph_exe_ctx)));
}

}  // namespace imperative
}  // namespace paddle
