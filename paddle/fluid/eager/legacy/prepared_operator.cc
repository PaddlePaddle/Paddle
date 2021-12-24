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

#include "paddle/fluid/eager/legacy/prepared_operator.h"

#include "paddle/fluid/eager/legacy/infer_shape_context.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/utils/small_vector.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif
DECLARE_bool(check_nan_inf);
DECLARE_bool(run_pten_kernel);

namespace egr {
namespace legacy {

const paddle::framework::Tensor* GetTensorFromVar(
    const paddle::framework::Variable& var) {
  if (var.IsType<paddle::framework::LoDTensor>()) {
    return &(var.Get<paddle::framework::LoDTensor>());
  } else if (var.IsType<paddle::framework::SelectedRows>()) {
    return &(var.Get<paddle::framework::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

static const paddle::framework::Attribute& GetAttr(
    const paddle::framework::AttributeMap& attrs,
    const paddle::framework::AttributeMap& default_attrs,
    const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  PADDLE_ENFORCE_EQ(found, true,
                    paddle::platform::errors::NotFound(
                        "(%s) is not found in AttributeMap.", name));
  return it->second;
}

static void HandleComplexGradToRealGrad(const NameTensorMap& outs) {
  // TODO(jiabin): Support complex forward datatype later.
}

PreparedOp::PreparedOp(
    const paddle::framework::OperatorBase& op,
    const paddle::framework::RuntimeContext& ctx,
    const paddle::framework::OpKernelType& kernel_type,
    const paddle::framework::OperatorWithKernel::OpKernelFunc& func,
    paddle::platform::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_type_(kernel_type),
      func_(func),
      dev_ctx_(dev_ctx) {}

PreparedOp PrepareImpl(const NameTensorMap& ins, const NameTensorMap& outs,
                       const paddle::framework::OperatorWithKernel& op,
                       const paddle::platform::Place& place,
                       const paddle::framework::AttributeMap& attrs,
                       const paddle::framework::AttributeMap& default_attrs) {
  VLOG(6) << "Preparing an Op";
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  paddle::framework::RuntimeContext ctx({}, {});

#ifdef PADDLE_WITH_MKLDNN
  // MKLDNN variant of code reads attributes in some of GetKernelTypeForVar and
  // GetKernelType functions, so we need to copy the attributes there.
  // Const qualifier of Attrs had to be discarded to overwrite it.
  if (FLAGS_use_mkldnn) {
    auto& mutable_op_attrs =
        const_cast<paddle::framework::AttributeMap&>(op.Attrs());
    mutable_op_attrs = default_attrs;
    for (auto& attr : attrs) {
      mutable_op_attrs[attr.first] = attr.second;
    }
  }
#endif

  // 1. get expected kernel key
  auto dygraph_exe_ctx = egr::legacy::EagerExecutionContext(
      op, paddle::framework::Scope(), *dev_ctx, ctx, ins, outs, attrs,
      default_attrs);
  auto expected_kernel_key = op.GetExpectedKernelType(dygraph_exe_ctx);
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  // 2. check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      paddle::platform::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));

  auto& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_XPU
  if (is_xpu_place(expected_kernel_key.place_) &&
      (kernel_iter == kernels.end() ||
       !paddle::platform::is_xpu_support_op(op.Type(), expected_kernel_key) ||
       paddle::platform::is_in_xpu_black_list(op.Type()))) {
    VLOG(3) << "missing XPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = paddle::platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  if (kernel_iter == kernels.end() &&
      is_npu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing NPU kernel: " << op.Type()
            << ", expected_kernel_key:" << expected_kernel_key
            << ", fallbacking to CPU one!";
    expected_kernel_key.place_ = paddle::platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that
  // case
  PADDLE_ENFORCE_NE(kernel_iter, kernels.end(),
                    paddle::platform::errors::NotFound(
                        "Operator %s does not have kernel for %s.", op.Type(),
                        KernelTypeToString(expected_kernel_key)));

  if (!(expected_kernel_key.place_ == place)) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
  }
  VLOG(6) << "Construct Prepared Op";
  return PreparedOp(op, ctx, expected_kernel_key, kernel_iter->second, dev_ctx);
}

PreparedOp PreparedOp::Prepare(
    const NameTensorMap& ins, const NameTensorMap& outs,
    const paddle::framework::OperatorWithKernel& op,
    const paddle::platform::Place& place,
    const paddle::framework::AttributeMap& attrs,
    const paddle::framework::AttributeMap& default_attrs) {
  return PrepareImpl(ins, outs, op, place, attrs, default_attrs);
}

static void PreparedOpRunImpl(
    const paddle::framework::OperatorBase& op,
    const paddle::framework::RuntimeContext& ctx,
    const paddle::framework::OpKernelType& kernel_type,
    const paddle::framework::OperatorWithKernel::OpKernelFunc& func,
    paddle::platform::DeviceContext* dev_ctx, const NameTensorMap& ins,
    const NameTensorMap& outs, const paddle::framework::AttributeMap& attrs,
    const paddle::framework::AttributeMap& default_attrs) {
  // TODO(zjl): remove scope in dygraph
  VLOG(6) << "Runing Prepared Op";
  paddle::framework::Scope scope;

  EagerInferShapeContext infer_shape_ctx(&ins, &outs, &attrs, &default_attrs,
                                         op.Type());
  static_cast<const paddle::framework::OperatorWithKernel&>(op).InferShape(
      &infer_shape_ctx);

  func(EagerExecutionContext(op, scope, *dev_ctx, ctx, ins, outs, attrs,
                             default_attrs));

  if (FLAGS_check_nan_inf) {
    paddle::framework::details::CheckOpHasNanOrInfInEager<EagerTensor>(
        op.Type(), outs, dev_ctx->GetPlace());
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
  if (paddle::framework::IsComplexType(kernel_type.data_type_)) {
    HandleComplexGradToRealGrad(outs);
  }
  VLOG(6) << "Finish Runing Prepared Op";
}

void PreparedOp::Run(const NameTensorMap& ins, const NameTensorMap& outs,
                     const paddle::framework::AttributeMap& attrs,
                     const paddle::framework::AttributeMap& default_attrs) {
  PreparedOpRunImpl(op_, ctx_, kernel_type_, func_, dev_ctx_, ins, outs, attrs,
                    default_attrs);
}

std::shared_ptr<NameTensorMap> PrepareData(
    const paddle::framework::OperatorWithKernel& op, const NameTensorMap& ins,
    const paddle::framework::OpKernelType& expected_kernel_key) {
  std::shared_ptr<NameTensorMap> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& egr_tensor = name_pair.second[i];
      const auto* tensor = GetTensorFromVar(egr_tensor->Var());
      if (tensor && tensor->IsInitialized()) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          // TODO(jiabin): Support Cache later
          VLOG(3) << "Transform Variable " << egr_tensor->name() << " from "
                  << kernel_type_for_var << " to " << expected_kernel_key;
          paddle::framework::Tensor out;
          TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                        &out);
          if (NeedTransformDataType(kernel_type_for_var, expected_kernel_key)) {
            // To avoid NameVarMap copy construction overhead in general
            // scenarios, if inplace transformed, return original input
            // directly
            if (tmp_ins_ptr == nullptr) {
              tmp_ins_ptr = std::make_shared<NameTensorMap>(ins);
            }
            auto tmp_egr_tensor =
                std::make_shared<EagerTensor>(egr_tensor->name());
            SetTensorToVariable(egr_tensor->Var(), out,
                                tmp_egr_tensor->MutableVar());
            (*tmp_ins_ptr)[name_pair.first][i] = tmp_egr_tensor;
          } else {
            // if dtype is same, transform inplace will not change the
            // original
            // value, transform inplace to avoid multiple copy
            SetTensorToVariable(egr_tensor->Var(), out,
                                egr_tensor->MutableVar());
          }
        }
      }
    }
  }
  return tmp_ins_ptr;
}

}  // namespace legacy
}  // namespace egr
