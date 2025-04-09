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

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/denormal.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/utils/string/string_helper.h"

COMMON_DECLARE_bool(use_mkldnn);
COMMON_DECLARE_string(tracer_onednn_ops_on);
COMMON_DECLARE_string(tracer_onednn_ops_off);
COMMON_DECLARE_bool(use_stride_kernel);

namespace paddle::imperative {
thread_local std::string Tracer::python_stack_ = "";

thread_local bool Tracer::use_layout_autotune_ = false;

static thread_local std::shared_ptr<Tracer> g_current_tracer(nullptr);

static thread_local std::shared_ptr<AmpAttrs> g_current_amp_attrs =
    std::make_shared<AmpAttrs>();

static thread_local bool g_has_grad = true;

TEST_API void Tracer::DisableLayoutAutoTune() { use_layout_autotune_ = false; }
TEST_API void Tracer::EnableLayoutAutoTune() {
  use_layout_autotune_ = true;
  if (FLAGS_use_stride_kernel) {
    LOG(WARNING) << "When the layout_autotune policy is on, Paddle will turn "
                    "off the Stride policy. This will cause the input and "
                    "output of the Strided API no longer share memory, which "
                    "may cause problems with model accuracy.";
    FLAGS_use_stride_kernel = false;
  }
}

bool Tracer::UseLayoutAutoTune() {
#if defined(PADDLE_WITH_CUDA)
  if (phi::backends::gpu::TensorCoreAvailable()) {
    return use_layout_autotune_;
  }
#endif
  use_layout_autotune_ = false;
  return false;
}

TEST_API void Tracer::SetPythonStack(std::string stack_str) {
  python_stack_ = stack_str;
}
TEST_API std::string Tracer::GetPythonStack() { return python_stack_; }

const std::shared_ptr<Tracer>& GetCurrentTracer() { return g_current_tracer; }

TEST_API void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer) {
  g_current_tracer = tracer;
  VLOG(6) << "Set current tracer: " << g_current_tracer;
}

const std::shared_ptr<AmpAttrs>& GetCurrentAmpAttrs() {
  return g_current_amp_attrs;
}

void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& pair : outs) {
    for (const auto& var : pair.second) {
      // NOTE(zhiqiu): this happens when None output are passed from python
      // side. For example, fake_quantize_dequantize_moving_average_abs_max may
      // pass None OutAccum in eval mode.
      // It can be refined by generate several different pybind interface for
      // one operator with different function signature.
      if (var == nullptr) {
        VLOG(4) << pair.first << " is NULL";
        continue;
      }
      VLOG(6) << "Set output: " << var->Name()
              << "'s OverriddenStopGradient as " << generate_grad;
      var->InnerSetOverriddenStopGradient(generate_grad);
    }
  }
}

void IncreaseVarbaseReferenceCountUntilCopyComplete(
    const std::shared_ptr<imperative::VarBase>& var, const phi::Place& place) {
  // Note(zhiqiu): Follow the logic of TensorCopy to determine the place that we
  // need to add callback, see tensor_utils.cc:245
  auto place_ = phi::is_gpu_place(place) ? place : var->Place();

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
    const phi::Place& place) {
  // if not exists, create a new GarbageCollector at given place
  if (gcs_.count(place) == 0) {
    std::unique_ptr<framework::GarbageCollector> gc;
    if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc = std::make_unique<framework::DefaultStreamGarbageCollector>(place, 0);

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use CUDA device since it's not compiled with CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (phi::is_cuda_pinned_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc = std::make_unique<framework::CUDAPinnedGarbageCollector>(place, 0);

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use CUDAPinned device since it's not compiled with "
          "CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (phi::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
      gc = std::make_unique<framework::XPUGarbageCollector>(place, 0);
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use XPU device since it's not compiled with XPU,"
          "Please recompile or reinstall Paddle with XPU support."));
#endif
    } else if (phi::is_cpu_place(place)) {
      gc = std::make_unique<framework::CPUGarbageCollector>(place, 0);
      VLOG(10) << "Created GarbageCollector at " << place;
    } else if (phi::is_ipu_place(place)) {
#if defined(PADDLE_WITH_IPU)
      gc = std::make_unique<framework::IPUGarbageCollector>(place, 0);
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use IPU device since it's not compiled with IPU,"
          "Please recompile or reinstall Paddle with IPU support."));
#endif
    } else if (phi::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      if (framework::IsFastEagerDeletionModeEnabled()) {
        gc =
            std::make_unique<framework::CustomDeviceUnsafeFastGarbageCollector>(
                place, 0);
        VLOG(10) << "Created UnsafeFastGarbageCollector at " << place;
      } else {
        gc = std::make_unique<framework::CustomDefaultStreamGarbageCollector>(
            place, 0);
        VLOG(10) << "Created GarbageCollector at " << place;
      }
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use CustomDevice since it's not compiled with "
          "CustomDevice,"
          "Please recompile or reinstall Paddle with CustomDevice "
          "support."));
#endif
    } else {
      PADDLE_THROW(common::errors::PreconditionNotMet(
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
                     const phi::Place& place,
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
                         const phi::Place& place,
                         bool trace_backward,
                         const std::map<std::string, std::string>& inplace_map,
                         paddle::framework::AttributeMap* passed_default_attrs_,
                         bool use_default_attr_map) {
  phi::RecordEvent op_type_record_event(
      type, phi::TracerEventType::Operator, 1);
  platform::ScopedFlushDenormal flush;
  VLOG(4) << "Trace Op: " << type;
  if (FLAGS_use_mkldnn) {
    // if both lists are empty all ops are enabled (default for
    // FLAGS_use_mkldnn=1)
    // if ops_on list is not empty only ops from that list are enabled
    if (!FLAGS_tracer_onednn_ops_on.empty()) {
      auto is_on = FLAGS_tracer_onednn_ops_on.find(type) != std::string::npos;
      attrs["use_mkldnn"] = is_on;
    } else {
      // if ops_on list is empty all ops are enabled except types from off_list
      auto is_off = FLAGS_tracer_onednn_ops_off.find(type) != std::string::npos;
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
  if (GetCurrentAmpAttrs()->GetAmpLevel() == AmpLevel::O1) {
    if (GetCurrentAmpAttrs()->GetAmpPhiDtype() == phi::DataType::FLOAT16) {
      VLOG(5) << "Float16 Auto Mixed Precision O1 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          AutoCastInputs<VarType>(type, ins));
    } else if (GetCurrentAmpAttrs()->GetAmpPhiDtype() ==
               phi::DataType::BFLOAT16) {
      VLOG(5) << "BFloat16 Auto Mixed Precision O1 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          AutoCastBF16Inputs<VarType>(type, ins));
    }
  } else if (GetCurrentAmpAttrs()->GetAmpLevel() == AmpLevel::O2) {
    if (GetCurrentAmpAttrs()->GetAmpPhiDtype() == phi::DataType::FLOAT16) {
      VLOG(5) << "Float16 Auto Mixed Precision O2 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          CastPureFp16Inputs<VarType>(type, ins));
    } else if (GetCurrentAmpAttrs()->GetAmpPhiDtype() ==
               phi::DataType::BFLOAT16) {
      VLOG(5) << "BFloat16 Auto Mixed Precision O2 run operator: " << type;
      ins_amp = std::make_unique<NameVarMap<VarType>>(
          CastPureBf16Inputs<VarType>(type, ins));
    }
  }

  if (phi::is_gpu_place(place)) {
    const auto& new_tmp = ins_amp == nullptr ? ins : *ins_amp;
    const auto& tracer = imperative::GetCurrentTracer();
    ins_amp = std::make_unique<NameVarMap<VarType>>(
        imperative::AutoTuneLayout<VarType>(
            type, new_tmp, outs, &attrs, tracer));
  }

  const auto& new_ins = ins_amp == nullptr ? ins : *ins_amp;

  try {
    if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::SetDeviceId(place.device);
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
      platform::SetXPUDeviceId(place.device);
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      phi::DeviceManager::SetDevice(place);
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "PaddlePaddle should compile with CustomDevice if use "
          "CustomPlace."));
#endif
    }

    if (!use_default_attr_map) {
      PADDLE_ENFORCE_NOT_NULL(passed_default_attrs_,
                              common::errors::PermissionDenied(
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
    throw exception;
  } catch (std::exception& ex) {
    PADDLE_THROW(
        common::errors::Fatal("Operator %s raises an %s exception.\n"
                              "The exception content is\n:%s.",
                              type,
                              common::demangle(typeid(ex).name()),
                              ex.what()));
  } catch (...) {
    // NOTE: this branch represents a very serious bug with
    // low probability of occurrence, and we can't get its
    // exception content here.
    PADDLE_THROW(common::errors::Fatal(
        "Operator %s raises an unknown exception.", type));
  }

  {
    phi::RecordEvent node_creation_record_event(
        "grad_node_creation", phi::TracerEventType::OperatorInner, 1);

    if (ComputeRequiredGrad(new_ins, outs, trace_backward)) {
      PADDLE_ENFORCE_EQ(
          passed_default_attrs_,
          nullptr,
          common::errors::PermissionDenied(
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

template TEST_API void Tracer::TraceOp<VarBase>(
    const std::string& type,
    const NameVarMap<VarBase>& ins,
    const NameVarMap<VarBase>& outs,
    framework::AttributeMap attrs,
    const phi::Place& place,
    bool trace_backward,
    const std::map<std::string, std::string>& inplace_map,
    paddle::framework::AttributeMap* default_attrs,
    bool use_default_attr_map);

template void Tracer::TraceOp<egr::EagerVariable>(
    const std::string& type,
    const NameVarMap<egr::EagerVariable>& ins,
    const NameVarMap<egr::EagerVariable>& outs,
    framework::AttributeMap attrs,
    const phi::Place& place,
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
                   g_has_grad,
                   inplace_map);
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap& attrs,
                     const phi::Place& place,
                     paddle::framework::AttributeMap* default_attrs,
                     bool use_default_attr_map,
                     const std::map<std::string, std::string>& inplace_map) {
  VLOG(6) << "Running On Eager TraceOp with use_default_attr_map: "
          << use_default_attr_map;
  std::map<phi::DenseTensor*, phi::DenseTensor*> need_backup_inputs2outputs;
  std::map<phi::DenseTensor*, std::shared_ptr<phi::Allocation>>
      need_backup_inputs2holder;
  std::map<phi::DenseTensor*, phi::DDim> need_backup_inputs2strides;
  std::map<phi::DenseTensor*, size_t> need_backup_inputs2offset;
  if (FLAGS_use_stride_kernel) {
    for (auto& iter : inplace_map) {
      auto inputs_iter = ins.find(iter.first);
      for (size_t i = 0; i < inputs_iter->second.size(); i++) {
        auto var = inputs_iter->second[i]->MutableVar();
        if (var->IsType<phi::DenseTensor>()) {
          auto dense_tensor = var->GetMutable<phi::DenseTensor>();
          if (!dense_tensor->meta().is_contiguous()) {
            NameTensorMap* tmp_out = const_cast<NameTensorMap*>(&outs);
            auto outputs_iter = tmp_out->find(iter.second);
            outputs_iter->second[i] = std::make_shared<egr::EagerVariable>(
                egr::Controller::Instance().GenerateUniqueName());
            need_backup_inputs2outputs[dense_tensor] =
                outputs_iter->second[i]
                    ->MutableVar()
                    ->GetMutable<phi::DenseTensor>();
            need_backup_inputs2holder[dense_tensor] = dense_tensor->Holder();
            need_backup_inputs2strides[dense_tensor] = dense_tensor->strides();
            need_backup_inputs2offset[dense_tensor] = dense_tensor->offset();
          }
        }
      }
    }
    TraceOpImpl<egr::EagerVariable>(type,
                                    ins,
                                    outs,
                                    attrs,
                                    place,
                                    false,
                                    {},
                                    default_attrs,
                                    use_default_attr_map);

    auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    for (auto& iter : need_backup_inputs2outputs) {
      iter.first->ResetHolder(need_backup_inputs2holder[iter.first]);
      iter.first->set_strides(need_backup_inputs2strides[iter.first]);
      iter.first->set_offset(need_backup_inputs2offset[iter.first]);
      paddle::experimental::TransStrideLegacy(dev_ctx, iter.second, iter.first);
      iter.second->ResetHolder(need_backup_inputs2holder[iter.first]);
      iter.second->set_strides(need_backup_inputs2strides[iter.first]);
      iter.second->set_offset(need_backup_inputs2offset[iter.first]);
    }
  } else {
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
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap attrs) {
  VLOG(6) << "Running On Eager TraceOp(4 args): ";
  TraceOpImpl<egr::EagerVariable>(
      type, ins, outs, attrs, expected_place_, false, {}, nullptr, true);
}

void Tracer::TraceOp(const std::string& type,
                     const NameTensorMap& ins,
                     const NameTensorMap& outs,
                     paddle::framework::AttributeMap& attrs,
                     const std::map<std::string, std::string>& inplace_map) {
  VLOG(6) << "Running On Eager TraceOp(less): ";

  std::map<phi::DenseTensor*, phi::DenseTensor*> need_backup_inputs2outputs;

  if (FLAGS_use_stride_kernel) {
    for (auto& iter : inplace_map) {
      auto inputs_iter = ins.find(iter.first);
      for (size_t i = 0; i < inputs_iter->second.size(); i++) {
        auto var = inputs_iter->second[i]->MutableVar();
        if (var->IsType<phi::DenseTensor>()) {
          auto dense_tensor = var->GetMutable<phi::DenseTensor>();
          if (!dense_tensor->meta().is_contiguous()) {
            NameTensorMap* tmp_out = const_cast<NameTensorMap*>(&outs);
            auto outputs_iter = tmp_out->find(iter.second);
            outputs_iter->second[i] = std::make_shared<egr::EagerVariable>(
                egr::Controller::Instance().GenerateUniqueName());
            need_backup_inputs2outputs[dense_tensor] =
                outputs_iter->second[i]
                    ->MutableVar()
                    ->GetMutable<phi::DenseTensor>();
          }
        }
      }
    }
  } else {
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
}

TEST_API void Tracer::SetExpectedPlace(phi::Place place) {
  expected_place_ = place;
}
TEST_API bool Tracer::HasGrad() const { return g_has_grad; }

TEST_API void Tracer::SetHasGrad(bool has_grad) { g_has_grad = has_grad; }

TEST_API void Tracer::SetUsePromote(bool use_promote) {
  VLOG(4) << "set use_promote to " << use_promote;
  g_current_amp_attrs->SetUsePromote(use_promote);
}

TEST_API bool Tracer::GetUsePromote() const {
  return g_current_amp_attrs->GetUsePromote();
}

TEST_API void Tracer::SetAmpLevel(AmpLevel level) {
  VLOG(4) << "set amp_level to " << static_cast<unsigned int>(level);
  g_current_amp_attrs->SetAmpLevel(level);
}

TEST_API AmpLevel Tracer::GetAmpLevel() const {
  return g_current_amp_attrs->GetAmpLevel();
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap& outs,
                                 bool trace_backward) {
  if (!trace_backward) return false;

  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverriddenStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverriddenStopGradient());
        return true;
      }
    }
  }
  return false;
}

void Tracer::SetAmpDtype(std::string amp_dtype) {
  VLOG(4) << "set amp_dtype to " << amp_dtype;
  g_current_amp_attrs->SetAmpDtype(amp_dtype);
}

std::string Tracer::GetAmpDtype() const {
  return g_current_amp_attrs->GetAmpDtype();
}

phi::DataType Tracer::GetAmpPhiDtype() const {
  return g_current_amp_attrs->GetAmpPhiDtype();
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
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
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
                    common::errors::InvalidArgument(
                        "This op type:`%s` is not a OperatorWithKernel, only "
                        "OperatorWithKernel can get KernelSignature",
                        type));
  if (phi::KernelFactory::Instance().HasStructuredKernel(type)) {
    return phi::KernelSignature(op->Type().c_str());
  } else {
    return phi::KernelSignature(
        opbase_with_kernel->GetExpectedPhiKernelArgs(dygraph_exe_ctx));
  }
}

}  // namespace paddle::imperative
