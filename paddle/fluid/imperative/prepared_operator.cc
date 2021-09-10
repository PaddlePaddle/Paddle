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

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/tcmpt_utils.h"
#include "paddle/fluid/imperative/infer_shape_context.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/xpu/xpu_op_list.h"
#endif
DECLARE_bool(check_nan_inf);

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
  } else if (var.IsType<framework::SelectedRows>()) {
    return &(var.Get<framework::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

template <typename T>
static const T& GetAttr(const framework::AttributeMap& attrs,
                        const framework::AttributeMap& default_attrs,
                        const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  PADDLE_ENFORCE_EQ(
      found, true,
      platform::errors::NotFound("(%s) is not found in AttributeMap.", name));
  return BOOST_GET_CONST(T, it->second);
}

template <typename VarType>
static void HandleComplexGradToRealGrad(const NameVarMap<VarType>& outs) {
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
                       const pt::KernelKey& pt_kernel_key,
                       const pt::Kernel& pt_kernel,
                       platform::DeviceContext* dev_ctx)
    : op_(op),
      ctx_(ctx),
      kernel_type_(framework::OpKernelType(framework::proto::VarType::RAW,
                                           platform::CPUPlace())),
      func_(nullptr),
      dev_ctx_(dev_ctx),
      run_pt_kernel_(true),
      pt_kernel_key_(pt_kernel_key),
      pt_kernel_(pt_kernel) {
  // TODO(chenweihang): PrepareData still use old impl, so here need save
  // old kernel type, trans it later
  kernel_type_ = framework::TransPtKernelKeyToOpKernelType(pt_kernel_key_);
}

template <typename VarType>
static framework::VariableValueMap BuildInputMap(
    const NameVarMap<VarType>& ins) {
  framework::VariableValueMap inputs;
  for (auto& var_pair : ins) {
    for (auto& var : var_pair.second) {
      inputs[var_pair.first].emplace_back(var->MutableVar());
    }
  }
  return inputs;
}

template <typename VarType>
bool ContainSelectedRows(const NameVarMap<VarType>& inputs) {
  for (auto& var_pair : inputs) {
    for (auto& var : var_pair.second) {
      if (var->Var().template IsType<framework::SelectedRows>()) {
        return true;
      }
    }
  }
  return false;
}

// TODO(chenweihang): enhance rules, not all dispensable inputs
// are host tensor, now only for scale kernel verify
template <typename VarType>
bool ContainHostTensor(const framework::proto::OpProto& op_proto,
                       const NameVarMap<VarType>& inputs) {
  for (int i = 0; i < op_proto.inputs_size(); ++i) {
    auto in = op_proto.inputs()[i];
    auto it = inputs.find(in.name());
    if (it == inputs.end()) {
      return false;
    }
    return it->second.empty() ? false : true;
  }
  return false;
}

template <typename VarType>
static pt::KernelName ConstructPtKernelName(
    const std::string& op_type, const framework::proto::OpProto& op_proto,
    const NameVarMap<VarType>& inputs) {
  std::string overload_name;
  if (ContainSelectedRows<VarType>(inputs)) {
    overload_name = pt::kContainSelectedRowsSuffix;
  }
  if (ContainHostTensor<VarType>(op_proto, inputs)) {
    if (overload_name != "") {
      overload_name += ".";
    }
    overload_name += pt::kContainHostTensorSuffix;
  }
  return pt::KernelName(op_type, overload_name);
}

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

  // 1. get expected kernel key
  if (pt::KernelFactory::Instance().ContainsKernel(op.Type().c_str())) {
    auto kernel_name =
        ConstructPtKernelName<VarType>(op.Type(), (*op.Info().proto_), ins);
    auto inputs = BuildInputMap<VarType>(ins);
    auto pt_kernel_key = op.ConstructPtKernelKey(inputs, place);
    auto pt_kernel =
        pt::KernelFactory::Instance().SelectKernel(kernel_name, pt_kernel_key);
    // for debug
    VLOG(1) << "PrepareImpl - kernel name: " << kernel_name
            << " | kernel key: " << pt_kernel_key << " | kernel: " << pt_kernel;
    if (pt_kernel.IsValid()) {
      // TODO(chenweihang): using CPUKernel when miss device kernel case
      return PreparedOp(op, ctx, pt_kernel_key, pt_kernel, dev_ctx);
    }
  }

  auto expected_kernel_key = op.GetExpectedKernelType(
      DygraphExecutionContext<VarType>(op, framework::Scope(), *dev_ctx, ctx,
                                       ins, outs, attrs, default_attrs));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  // 2. check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::NotFound(
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
    expected_kernel_key.place_ = platform::CPUPlace();
    kernel_iter = kernels.find(expected_kernel_key);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  if (kernel_iter == kernels.end() &&
      is_npu_place(expected_kernel_key.place_)) {
    VLOG(3) << "missing NPU kernel: " << op.Type()
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

template <typename VarType>
static pt::KernelContext BuildDygraphKernelContext(
    const pt::Kernel& pt_kernel, const framework::proto::OpProto& op_proto,
    const NameVarMap<VarType>& ins, const NameVarMap<VarType>& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const platform::DeviceContext& dev_ctx) {
  // TODO(chenweihang): now only work for very simple case (sign op),
  // many cases need to be deal with later:
  // 1. the input and output are not tensor
  // 2. the dispensbale, duplicable input and output
  // 3. needless attributes remove
  // 4. use pt Tensor directly
  // 5. kernel input is not DenseTensor
  pt::KernelContext op_kernel_ctx(dev_ctx);
  auto input_defs = pt_kernel.args_def().input_defs();
  auto output_defs = pt_kernel.args_def().output_defs();

  // TODO(chenweihang): For scale op, when the input has a `ScaleTensor`,
  // the following scale attribute should be skipped, and there are many
  // such ops, which require certain rules to process, now only for verify
  // scale op
  std::unordered_map<std::string, bool> contain_host_tensor_flags{
      {"ScaleTensor", false}};
  std::unordered_map<std::string, std::string> attr_to_host_tensor{
      {"scale", "ScaleTensor"}};

  for (int i = 0; i < op_proto.inputs_size(); ++i) {
    auto in = op_proto.inputs()[i];
    // TODO(chenweihang): deal with diff param in vector
    if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
      VLOG(1) << "Dygraph PtKernel input: skip extra & quant input - "
              << in.name();
      continue;
    }
    auto in_name = in.name();
    if (in.has_dispensable() && in.dispensable()) {
      if (contain_host_tensor_flags.count(in_name) > 0 &&
          ins.count(in_name) > 0 && ins.at(in_name).size() > 0) {
        VLOG(1) << "Dygraph PtKernel input: contain host input - " << in_name;
        contain_host_tensor_flags[in_name] = true;
      } else {
        VLOG(1) << "Dygraph PtKernel input: skip dispensable input - "
                << in_name;
        continue;
      }
    }
    VLOG(1) << "Dygraph PtKernel input: " << in_name;
    auto in_def = input_defs.at(i);
    for (auto var : ins.at(in_name)) {
      const auto& variable = var->Var();
      if (variable.template IsType<framework::LoDTensor>()) {
        const auto& tensor = variable.template Get<framework::LoDTensor>();
        auto pt_in =
            framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
                tensor, in_def.backend, in_def.dtype, in_def.layout);
        op_kernel_ctx.EmplaceBackInput(pt_in);
      } else if (variable.template IsType<framework::SelectedRows>()) {
        const auto& tensor = variable.template Get<framework::SelectedRows>();
        auto pt_in = framework::MakeTensorImpl<pt::SelectedRowsTensor,
                                               framework::SelectedRows>(
            tensor, in_def.backend, in_def.dtype, in_def.layout);
        op_kernel_ctx.EmplaceBackInput(pt_in);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported shared input `%s` type now when call pt kernel.",
            framework::ToTypeName(variable.Type())));
      }
    }
  }

  for (int i = 0; i < op_proto.outputs_size(); ++i) {
    auto out_name = op_proto.outputs()[i].name();
    VLOG(1) << "Dygraph PtKernel output: " << out_name;
    // TODO(chenweihang): outputs also need skip some cases
    auto out_def = output_defs.at(i);
    for (auto var : outs.at(out_name)) {
      // mutable_data before run kernel, to avoid share output form
      // KernelContext to original tensor
      auto* variable = var->MutableVar();
      if (variable->template IsType<framework::LoDTensor>()) {
        auto* tensor = variable->template GetMutable<framework::LoDTensor>();
        tensor->mutable_data(pt::TransToFluidPlace(out_def.backend),
                             pt::TransToProtoVarType(out_def.dtype));
        auto pt_out =
            framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
                *tensor, out_def.backend, out_def.dtype, out_def.layout);
        op_kernel_ctx.EmplaceBackOutput(pt_out);
      } else if (variable->template IsType<framework::SelectedRows>()) {
        auto* tensor = variable->template GetMutable<framework::SelectedRows>();
        tensor->mutable_value()->mutable_data(
            pt::TransToFluidPlace(out_def.backend),
            pt::TransToProtoVarType(out_def.dtype));
        auto pt_out = framework::MakeTensorImpl<pt::SelectedRowsTensor,
                                                framework::SelectedRows>(
            *tensor, out_def.backend, out_def.dtype, out_def.layout);
        op_kernel_ctx.EmplaceBackOutput(pt_out);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported shared output `%s` type now when call pt kernel.",
            framework::ToTypeName(variable->Type())));
      }
    }
  }

  for (int i = 0; i < op_proto.attrs_size(); ++i) {
    auto attr = op_proto.attrs()[i];
    VLOG(1) << "Dygraph PtKernel attribute: " << attr.name();
    if (attr.name() == "use_mkldnn" || attr.name() == "op_role" ||
        attr.name() == "op_role_var" || attr.name() == "op_namescope" ||
        attr.name() == "op_callstack" || attr.name() == "op_device") {
      VLOG(1) << "Dygraph PtKernel attribute: skip needless attr - "
              << attr.name();
      continue;
    }
    if ((attr.has_extra() && attr.extra()) ||
        (attr.has_quant() && attr.quant())) {
      VLOG(1) << "Dygraph PtKernel attribute: skip extra & quant attr - "
              << attr.name();
      continue;
    }
    if (attr_to_host_tensor.count(attr.name()) > 0 &&
        contain_host_tensor_flags.at(attr_to_host_tensor.at(attr.name())) ==
            true) {
      VLOG(1) << "Dygraph PtKernel attribute: skip dynaimc attr - "
              << attr.name() << ", because "
              << attr_to_host_tensor.at(attr.name()) << " exists.";
      continue;
    }
    // TODO(chenweihang): support other attrs
    // In principle, the attr required by the dynamic mode should be
    // passed in from the Python side, and there is no need to look up
    // from the default_map, but now this nor work
    switch (attr.type()) {
      case framework::proto::AttrType::INT:
        op_kernel_ctx.EmplaceBackAttr(
            GetAttr<int>(attrs, default_attrs, attr.name()));
        break;
      case framework::proto::AttrType::FLOAT:
        op_kernel_ctx.EmplaceBackAttr(
            GetAttr<float>(attrs, default_attrs, attr.name()));
        break;
      case framework::proto::AttrType::BOOLEAN:
        op_kernel_ctx.EmplaceBackAttr(
            GetAttr<bool>(attrs, default_attrs, attr.name()));
        break;
      default:
        // TODO(chenweihang): support other attrs type
        PADDLE_THROW(platform::errors::Unimplemented(
            "unsupported cast op attribute `%s` when construct "
            "KernelContext.",
            attr.name()));
    }
  }

  return op_kernel_ctx;
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

  DygraphInferShapeContext<VarType> infer_shape_ctx(&ins, &outs, &attrs,
                                                    &default_attrs, op.Type());
  static_cast<const framework::OperatorWithKernel&>(op).InferShape(
      &infer_shape_ctx);

  func(DygraphExecutionContext<VarType>(op, scope, *dev_ctx, ctx, ins, outs,
                                        attrs, default_attrs));

  if (FLAGS_check_nan_inf) {
    framework::details::CheckOpHasNanOrInfInDygraph<VarType>(
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
  if (framework::IsComplexType(kernel_type.data_type_)) {
    HandleComplexGradToRealGrad<VarType>(outs);
  }
}

template <typename VarType>
static void PreparedOpRunPtImpl(const framework::OperatorBase& op,
                                const pt::KernelKey& pt_kernel_key,
                                const pt::Kernel& pt_kernel,
                                platform::DeviceContext* dev_ctx,
                                const NameVarMap<VarType>& ins,
                                const NameVarMap<VarType>& outs,
                                const framework::AttributeMap& attrs,
                                const framework::AttributeMap& default_attrs) {
  DygraphInferShapeContext<VarType> infer_shape_ctx(&ins, &outs, &attrs,
                                                    &default_attrs, op.Type());
  static_cast<const framework::OperatorWithKernel&>(op).InferShape(
      &infer_shape_ctx);

  auto op_kernel_ctx =
      BuildDygraphKernelContext<VarType>(pt_kernel, *(op.Info().proto_), ins,
                                         outs, attrs, default_attrs, *dev_ctx);
  pt_kernel(&op_kernel_ctx);

  // TODO(chenweihang): add flags
  // TODO(chenweihang): deal with complex cases
}

void PreparedOp::Run(const NameVarMap<VarBase>& ins,
                     const NameVarMap<VarBase>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_pt_kernel_) {
    PreparedOpRunPtImpl<VarBase>(op_, pt_kernel_key_, pt_kernel_, dev_ctx_, ins,
                                 outs, attrs, default_attrs);
  } else {
    PreparedOpRunImpl<VarBase>(op_, ctx_, kernel_type_, func_, dev_ctx_, ins,
                               outs, attrs, default_attrs);
  }
}

void PreparedOp::Run(const NameVarMap<VariableWrapper>& ins,
                     const NameVarMap<VariableWrapper>& outs,
                     const framework::AttributeMap& attrs,
                     const framework::AttributeMap& default_attrs) {
  if (run_pt_kernel_) {
    PreparedOpRunPtImpl<VariableWrapper>(op_, pt_kernel_key_, pt_kernel_,
                                         dev_ctx_, ins, outs, attrs,
                                         default_attrs);
  } else {
    PreparedOpRunImpl<VariableWrapper>(op_, ctx_, kernel_type_, func_, dev_ctx_,
                                       ins, outs, attrs, default_attrs);
  }
}

}  // namespace imperative
}  // namespace paddle
