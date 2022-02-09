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

#include "paddle/fluid/imperative/prepared_operator.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/utils/small_vector.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif
#include "paddle/fluid/framework/library_type.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(check_nan_inf);
DECLARE_bool(benchmark);
DECLARE_bool(run_kp_kernel);

namespace paddle {
namespace imperative {

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<paddle::imperative::VarBase>& var) {
  return var->SharedVar();
}

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VariableWrapper>& var) {
  return var;
}

const framework::Tensor* GetTensorFromVar(const framework::Variable& var) {
  if (var.IsType<framework::LoDTensor>()) {
    return &(var.Get<framework::LoDTensor>());
  } else if (var.IsType<pten::SelectedRows>()) {
    return &(var.Get<pten::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

template <typename VarType>
void HandleComplexGradToRealGrad(const NameVarMap<VarType>& outs) {
  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      if (var == nullptr) {
        continue;
      }
      if (var->ForwardDataType() ==
          static_cast<framework::proto::VarType::Type>(-1)) {
        VLOG(6) << "Var (" << var->Name()
                << ")'s forward data type is not set.";
        continue;
      }
      if (!framework::IsComplexType(var->DataType()) ||
          framework::IsComplexType(var->ForwardDataType())) {
        continue;
      }
      const auto* tensor = GetTensorFromVar(var->Var());
      if (tensor && tensor->IsInitialized()) {
        VLOG(6) << "Transform " << framework::DataTypeToString(var->DataType())
                << " var `" << var->Name() << "` to "
                << framework::DataTypeToString(var->ForwardDataType())
                << " real var in dynamic graph.";
        framework::Tensor out;
        framework::TransComplexToReal(var->ForwardDataType(), var->DataType(),
                                      *tensor, &out);
        SetTensorToVariable(var->Var(), out, var->MutableVar());
      }
    }
  }
}

template <>
void HandleComplexGradToRealGrad<egr::EagerTensor>(
    const NameVarMap<egr::EagerTensor>& outs) {
  // TODO(jiabin): Support Complex here.
}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const framework::OpKernelType& kernel_type,
                       const framework::OperatorWithKernel::OpKernelFunc& func,
                       platform::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_type_(kernel_type),
      func_(func),
      dev_ctx_(dev_ctx) {}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       const framework::OpKernelType& kernel_type,
                       const framework::KernelSignature& kernel_signature,
                       const pten::Kernel& pt_kernel,
                       platform::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_type_(kernel_type),
      func_(nullptr),
      dev_ctx_(dev_ctx),
      run_pten_kernel_(true),
      pt_kernel_signature_(kernel_signature),
      pt_kernel_(pt_kernel) {}

template <typename VarType>
PreparedOp PrepareImpl(const NameVarMap<VarType>& ins,
                       const NameVarMap<VarType>& outs,
                       const framework::OperatorWithKernel& op,
                       const platform::Place& place,
                       const framework::AttributeMap& attrs,
                       const framework::AttributeMap& default_attrs) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  framework::RuntimeContext ctx({}, {});

#ifdef PADDLE_WITH_MKLDNN
  // MKLDNN variant of code reads attributes in some of GetKernelTypeForVar and
  // GetKernelType functions, so we need to copy the attributes there.
  // Const qualifier of Attrs had to be discarded to overwrite it.
  if (FLAGS_use_mkldnn) {
    auto& mutable_op_attrs = const_cast<framework::AttributeMap&>(op.Attrs());
    mutable_op_attrs = default_attrs;
    for (auto& attr : attrs) {
      mutable_op_attrs[attr.first] = attr.second;
    }
  }
#endif
  // NOTE(zhiqiu): for kernels on given device, for example NPU, the order to
  // choose is:
  // pten npu kernel > fluid npu kernel > pten cpu kernel > fluid cpu kernel

  // 1. get expected kernel key
  auto dygraph_exe_ctx = DygraphExecutionContext<VarType>(
      op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs, default_attrs);
  auto expected_kernel_key = op.GetExpectedKernelType(dygraph_exe_ctx);

  framework::KernelSignature pt_kernel_signature;
  pten::KernelKey pt_kernel_key;
  std::string pt_kernel_name;
  if (pten::KernelFactory::Instance().HasCompatiblePtenKernel(op.Type())) {
    pt_kernel_signature = op.GetExpectedPtenKernelArgs(dygraph_exe_ctx);
    VLOG(6) << pt_kernel_signature;

    pt_kernel_name = pt_kernel_signature.name;
    pt_kernel_key = TransOpKernelTypeToPtenKernelKey(expected_kernel_key);
    auto pt_kernel = pten::KernelFactory::Instance().SelectKernel(
        pt_kernel_name, pt_kernel_key);

    if (pt_kernel.IsValid()) {
      VLOG(6) << "Dynamic mode PrepareImpl - kernel name: " << pt_kernel_name
              << " | kernel key: " << pt_kernel_key
              << " | kernel: " << pt_kernel;

      if (platform::is_cpu_place(expected_kernel_key.place_)) {
        auto* cpu_ctx = pool.Get(paddle::platform::CPUPlace());
        return PreparedOp(op, ctx, expected_kernel_key, pt_kernel_signature,
                          pt_kernel, cpu_ctx);
      }
      // TODO(chenweihang): using CPUKernel when miss device kernel case
      return PreparedOp(op, ctx, expected_kernel_key, pt_kernel_signature,
                        pt_kernel, dev_ctx);
    } else {
      VLOG(6) << "Dynamic mode ChoosePtenKernel - kernel `" << pt_kernel_name
              << "` not found.";
    }
  }

  // 2. check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());

  if ((kernels_iter == all_op_kernels.end() ||
       kernels_iter->second.find(expected_kernel_key) ==
           kernels_iter->second.end())
#ifdef PADDLE_WITH_XPU
      ||
      paddle::platform::is_xpu_place(expected_kernel_key.place_) &&
          !paddle::platform::is_xpu_support_op(op.Type(),
                                               expected_kernel_key) ||
      paddle::platform::is_in_xpu_black_list(op.Type())
#endif
          ) {
    if (pten::KernelFactory::Instance().HasCompatiblePtenKernel(op.Type())) {
      auto pt_cpu_kernel_key =
          FallBackToCpu(expected_kernel_key, pt_kernel_key, op);
      auto pt_cpu_kernel = pten::KernelFactory::Instance().SelectKernel(
          pt_kernel_name, pt_cpu_kernel_key);
      if (pt_cpu_kernel.IsValid()) {
        VLOG(6) << "Dynamic mode PrepareImpl - kernel name: " << pt_kernel_name
                << " | kernel key: " << pt_cpu_kernel_key
                << " | kernel: " << pt_cpu_kernel;
        auto* cpu_ctx = pool.Get(paddle::platform::CPUPlace());
        return PreparedOp(op, ctx, expected_kernel_key, pt_kernel_signature,
                          pt_cpu_kernel, cpu_ctx);
      }
    }
  }

  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));
  auto& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(expected_kernel_key);

#ifdef PADDLE_WITH_XPU
  if (paddle::platform::is_xpu_place(expected_kernel_key.place_) &&
      (kernel_iter == kernels.end() ||
       !paddle::platform::is_xpu_support_op(op.Type(), expected_kernel_key) ||
       paddle::platform::is_in_xpu_black_list(op.Type()))) {
    VLOG(3) << "missing XPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }

#endif

#ifdef PADDLE_WITH_XPU_KP
  bool use_xpu_kp_kernel_rt =
      FLAGS_run_kp_kernel &&
      paddle::platform::is_xpu_kp_support_op(op.Type(), expected_kernel_key);
  bool use_xpu_kp_kernel_debug =
      paddle::platform::is_in_xpu_kpwhite_list(op.Type());
  if (use_xpu_kp_kernel_rt) {
    VLOG(3) << "xpu_kp using rt mode ";
  }
  if (use_xpu_kp_kernel_debug) {
    VLOG(3) << "xpu_kp using debug mode ";
  }
  if (paddle::platform::is_xpu_place(expected_kernel_key.place_) &&
      (use_xpu_kp_kernel_rt || use_xpu_kp_kernel_debug)) {
    expected_kernel_key.place_ = platform::XPUPlace();
    expected_kernel_key.library_type_ = paddle::framework::LibraryType::kKP;
    kernel_iter = kernels.find(expected_kernel_key);
    VLOG(3) << "using XPU KP kernel: " << op.Type()
            << ", using_kernel_key:" << expected_kernel_key;
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  if (kernel_iter == kernels.end() &&
      paddle::platform::is_npu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing NPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
#ifdef PADDLE_WITH_MLU
  if (kernel_iter == kernels.end() &&
      paddle::platform::is_mlu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing MLU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that
  // case
  PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                    platform::errors::NotFound(
                        "Operator %s does not have kernel for %s.", op.Type(),
                        KernelTypeToString(expected_kernel_key)));

  if (!(expected_kernel_key.place_ == place)) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
  }

  return PreparedOp(op, ctx, expected_kernel_key, kernel_iter->second, dev_ctx);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VarBase>& ins,
                               const NameVarMap<VarBase>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<VarBase>(ins, outs, op, place, attrs, default_attrs);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<VariableWrapper>& ins,
                               const NameVarMap<VariableWrapper>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<VariableWrapper>(ins, outs, op, place, attrs,
                                      default_attrs);
}

PreparedOp PreparedOp::Prepare(const NameVarMap<egr::EagerTensor>& ins,
                               const NameVarMap<egr::EagerTensor>& outs,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place,
                               const framework::AttributeMap& attrs,
                               const framework::AttributeMap& default_attrs) {
  return PrepareImpl<egr::EagerTensor>(ins, outs, op, place, attrs,
                                       default_attrs);
}
template <typename VarType>
static void PreparedOpRunImpl(
    const framework::OperatorBase& op, const framework::RuntimeContext& ctx,
    const framework::OpKernelType& kernel_type,
    const framework::OperatorWithKernel::OpKernelFunc& func,
    platform::DeviceContext* dev_ctx, const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs, const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs) {
  // TODO(zjl): remove scope in dygraph
  framework::Scope scope;

  {
    platform::RecordEvent record_event(op.Type() + " infer_shape",
                                       platform::EventRole::kInnerOp);
    DygraphInferShapeContext<VarType> infer_shape_ctx(
        &ins, &outs, &attrs, &default_attrs, op.Type(), &kernel_type);
    op.Info().infer_shape_(&infer_shape_ctx);
  }

  {
    platform::RecordEvent record_event(op.Type() + " compute",
                                       platform::EventRole::kInnerOp);

    func(DygraphExecutionContext<VarType>(op, scope, *dev_ctx, ctx, ins, outs,
                                          attrs, default_attrs));
  }

  if (FLAGS_check_nan_inf) {
    framework::details::CheckOpHasNanOrInfInDygraph<VarType>(
        op.Type(), outs, dev_ctx->GetPlace());
  }

  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op.Type() << "): context wait and get last error";
#endif
  }

  /**
   * [ Why need handle complex gradient to real gradient? ]
   *
   * After the introduction of complex number calculations, Ops that support
   * complex number calculations generally support type promotion, such as
   * x(float32) + y(complex64) = out(complex64), then the type of the grad
   * tensor should be dout(complex64), dx(float32), dy (complex64).
   *
   * But because the dout is complex64, the dx is also complex64 after
   * grad op kernel executed, we need to recognize this situation and
   * convert dx to float32 type. HandleComplexGradToRealGrad does this thing.
   */
  if (framework::IsComplexType(kernel_type.data_type_)) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

template <typename VarType>
static void PreparedOpRunPtImpl(
    const framework::OperatorBase& op,
    const framework::OpKernelType& kernel_type,
    const framework::KernelSignature& pt_kernel_signature,
    const pten::Kernel& pt_kernel, platform::DeviceContext* dev_ctx,
    const NameVarMap<VarType>& ins, const NameVarMap<VarType>& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs) {
  {
    platform::RecordEvent record_event(op.Type() + " infer_shape",
                                       platform::EventRole::kInnerOp);
    DygraphInferShapeContext<VarType> infer_shape_ctx(
        &ins, &outs, &attrs, &default_attrs, op.Type(), &kernel_type);
    op.Info().infer_shape_(&infer_shape_ctx);
  }

  {
    platform::RecordEvent record_event(op.Type() + " compute",
                                       platform::EventRole::kInnerOp);

    PreparePtenData<VarType>(pt_kernel, pt_kernel_signature, ins);

    pten::KernelContext pt_kernel_context;
    BuildDygraphPtenKernelContext<VarType>(pt_kernel_signature, pt_kernel, ins,
                                           outs, attrs, default_attrs, dev_ctx,
                                           &pt_kernel_context);

    pt_kernel(&pt_kernel_context);
  }

  if (FLAGS_benchmark) {
    dev_ctx->Wait();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    VLOG(4) << "Operator(" << op.Type() << "): context wait and get last error";
#endif
  }

  // TODO(chenweihang): add debug flags later
  if (framework::IsComplexType(kernel_type.data_type_)) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_pten_kernel_) {
    PreparedOpRunPtImpl<VarBase>(op_, kernel_type_, pt_kernel_signature_,
                                 pt_kernel_, dev_ctx_, ins, outs, attrs,
                                 default_attrs);
  } else {
    PreparedOpRunImpl<VarBase>(op_, ctx_, kernel_type_, func_, dev_ctx_, ins,
                               outs, attrs, default_attrs);
  }
}

void PreparedOp::Run(const NameVarMap<VariableWrapper>& ins,
                     const NameVarMap<VariableWrapper>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_pten_kernel_) {
    PreparedOpRunPtImpl<VariableWrapper>(
        op_, kernel_type_, pt_kernel_signature_, pt_kernel_, dev_ctx_, ins,
        outs, attrs, default_attrs);
  } else {
    PreparedOpRunImpl<VariableWrapper>(op_, ctx_, kernel_type_, func_, dev_ctx_,
                                       ins, outs, attrs, default_attrs);
  }
}

void PreparedOp::Run(const NameVarMap<egr::EagerTensor>& ins,
                     const NameVarMap<egr::EagerTensor>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_pten_kernel_) {
    PreparedOpRunPtImpl<egr::EagerTensor>(
        op_, kernel_type_, pt_kernel_signature_, pt_kernel_, dev_ctx_, ins,
        outs, attrs, default_attrs);
  } else {
    PreparedOpRunImpl<egr::EagerTensor>(op_, ctx_, kernel_type_, func_,
                                        dev_ctx_, ins, outs, attrs,
                                        default_attrs);
  }
}

}  // namespace imperative
}  // namespace paddle
