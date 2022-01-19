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
#include "paddle/fluid/imperative/infer_shape_context.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/utils/small_vector.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"
#endif
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

DECLARE_bool(check_nan_inf);
DECLARE_bool(run_pten_kernel);
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
  } else if (var.IsType<framework::SelectedRows>()) {
    return &(var.Get<framework::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

static const framework::Attribute& GetAttr(
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs, const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  PADDLE_ENFORCE_EQ(
      found, true,
      platform::errors::NotFound("(%s) is not found in AttributeMap.", name));
  return it->second;
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

  // 1. get expected kernel key
  auto dygraph_exe_ctx = DygraphExecutionContext<VarType>(
      op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs, default_attrs);
  auto expected_kernel_key = op.GetExpectedKernelType(dygraph_exe_ctx);
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  if (FLAGS_run_pten_kernel &&
      pten::KernelFactory::Instance().HasCompatiblePtenKernel(op.Type())) {
    auto pt_kernel_signature = op.GetExpectedPtenKernelArgs(dygraph_exe_ctx);
    VLOG(6) << pt_kernel_signature;

    auto pt_kernel_name = pt_kernel_signature.name;
    auto pt_kernel_key = TransOpKernelTypeToPtenKernelKey(expected_kernel_key);
    auto pt_kernel = pten::KernelFactory::Instance().SelectKernel(
        pt_kernel_name, pt_kernel_key);

    if (pt_kernel.IsValid()) {
      VLOG(6) << "Dynamic mode PrepareImpl - kernel name: " << pt_kernel_name
              << " | kernel key: " << pt_kernel_key
              << " | kernel: " << pt_kernel;

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

template <typename VarType>
void PreparePtenData(const pten::Kernel& pt_kernel,
                     const framework::KernelSignature& pt_kernel_signature,
                     const NameVarMap<VarType>& ins) {
  auto& input_names = std::get<0>(pt_kernel_signature.args);
  auto& input_defs = pt_kernel.args_def().input_defs();

  PADDLE_ENFORCE_EQ(input_names.size(), input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(), input_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& in_def = input_defs.at(i);
    auto& ins_vector = ins.at(input_names[i]);

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      auto var_base = ins_vector[offset];
      const auto* tensor_in = GetTensorFromVar(var_base->Var());
      if (tensor_in && tensor_in->IsInitialized()) {
        auto expected_place = pten::TransToFluidPlace(in_def.backend);
        if (platform::is_same_place(tensor_in->place(), expected_place)) {
          continue;
        }

        // TODO(zyfncg): Now there is no kernel which need to transform input
        // data, so we commented out following code temporarily,
        // and it will be used in the future.

        // VLOG(3) << "Pten Transform Variable " << var_base->Name() << " from "
        //         << tensor_in->place() << " to " << expected_place;

        // framework::Tensor tmp_tensor;
        // framework::TensorCopySync(*tensor_in, expected_place, &tmp_tensor);

        // SetTensorToVariable(var_base->Var(), tmp_tensor,
        //                     var_base->MutableVar());
      }
    }
  }
}

template <typename VarType>
static void BuildDygraphPtenKernelContext(
    const framework::KernelSignature& pt_kernel_signature,
    const pten::Kernel& pt_kernel, const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs, const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    platform::DeviceContext* dev_ctx, pten::KernelContext* kernel_ctx) {
  kernel_ctx->SetDeviceContext(dev_ctx);

  auto& input_names = std::get<0>(pt_kernel_signature.args);
  auto& attr_names = std::get<1>(pt_kernel_signature.args);
  auto& output_names = std::get<2>(pt_kernel_signature.args);

  auto& input_defs = pt_kernel.args_def().input_defs();
  auto& output_defs = pt_kernel.args_def().output_defs();
  auto& attr_defs = pt_kernel.args_def().attribute_defs();

  PADDLE_ENFORCE_EQ(input_names.size(), input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(), input_defs.size()));

  PADDLE_ENFORCE_EQ(output_names.size(), output_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of outputs_args names (%d) must be equal to "
                        "the size of kernel output_defs (%d).",
                        output_names.size(), output_defs.size()));

  PADDLE_ENFORCE_EQ(attr_names.size(), attr_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of attribute_args names (%d) must be equal "
                        "to the size of kernel attribute_defs (%d).",
                        attr_names.size(), attr_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& ins_vector = ins.at(input_names[i]);

    size_t start_idx = (i == 0 ? 0 : kernel_ctx->InputRangeAt(i - 1).second);
    size_t end_idx = start_idx + ins_vector.size();

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      const auto* tensor_in = GetTensorFromVar(ins_vector[offset]->Var());
      kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
    }
    kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
  }

  for (size_t i = 0; i < output_names.size(); ++i) {
    size_t start_idx = (i == 0 ? 0 : kernel_ctx->OutputRangeAt(i - 1).second);

    auto iter = outs.find(output_names[i]);
    if (iter == outs.end()) {
      kernel_ctx->EmplaceBackOutputWithoutSetRange({nullptr});
      kernel_ctx->AssignOutputRange(std::make_pair(start_idx, start_idx + 1),
                                    i);
      continue;
    }

    auto& outs_vector = iter->second;
    size_t end_idx = start_idx + outs_vector.size();

    for (size_t offset = 0; offset < outs_vector.size(); ++offset) {
      auto* var = outs_vector[offset]->MutableVar();
      framework::Tensor* tensor_out = nullptr;
      if (var->template IsType<framework::LoDTensor>()) {
        tensor_out = var->template GetMutable<framework::LoDTensor>();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported output `%s` type when call pt kernel.",
            framework::ToTypeName(var->Type())));
      }  // TODO(zyfncg): Add support for SelectedRows

      experimental::ResetTensorByArgDef(tensor_out, output_defs.at(i));
      framework::SetAllocationForOutputTenosr(
          tensor_out, pten::TransToFluidPlace(output_defs.at(i).backend));

      kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
    }
    kernel_ctx->AssignOutputRange(std::make_pair(start_idx, end_idx), i);
  }

  for (size_t i = 0; i < attr_names.size(); ++i) {
    if (attr_defs[i].type_index == std::type_index(typeid(pten::ScalarArray))) {
      if (attrs.find(attr_names[i]) !=
          attrs.end()) {  // shape is in the attribute
        auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int64_t>))) {
          kernel_ctx->EmplaceBackAttr(std::move(
              pten::ScalarArray(BOOST_GET_CONST(std::vector<int64_t>, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::vector<int32_t>))) {
          kernel_ctx->EmplaceBackAttr(std::move(
              pten::ScalarArray(BOOST_GET_CONST(std::vector<int32_t>, attr))));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to VectorTensor when "
              "construct KernelContext.",
              attr_names[i]));
        }
      } else {  // shape is in the input
        auto& ins_vector = ins.at(attr_names[i]);
        if (ins_vector.size() == 1) {  // ShapeTensor
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePtenScalarArrayFromVar(ins_vector[0]->Var())));
        } else {  // ShapeTensorList
          std::vector<framework::Variable*> variables;
          variables.reserve(ins_vector.size());
          for (const auto& var_base : ins_vector) {
            variables.push_back(var_base->MutableVar());
          }
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePtenScalarArrayFromVarList(variables)));
        }
      }
    } else if (attr_defs[i].type_index ==
               std::type_index(typeid(pten::Scalar))) {
      // TODO(chenweihang): support other attrs later
      // TODO(zhangyunfei): Scalar should hold scaler type, and we should check
      // attribtue type by attr_defs
      if (attrs.find(attr_names[i]) != attrs.end() ||
          default_attrs.find(attr_names[i]) !=
              default_attrs.end()) {  // scalar is in the attribute
        auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
        if (std::type_index(attr.type()) == std::type_index(typeid(float))) {
          kernel_ctx->EmplaceBackAttr(
              std::move(pten::Scalar(BOOST_GET_CONST(float, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::string))) {
          kernel_ctx->EmplaceBackAttr(
              std::move(pten::Scalar(BOOST_GET_CONST(std::string, attr))));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to Scalar when construct "
              "KernelContext in dygraph.",
              attr_names[i]));
        }
      } else {  // scalar is in the input
        auto& ins_vector = ins.at(attr_names[i]);
        kernel_ctx->EmplaceBackAttr(std::move(
            experimental::MakePtenScalarFromVar(ins_vector[0]->Var())));
      }

    } else {
      // TODO(chenweihang): support other attrs later
      auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
      if (attr_defs[i].type_index == std::type_index(typeid(int))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(int, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(float))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(float, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(bool))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(bool, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(pten::DataType))) {
        auto data_type = pten::TransToPtenDataType(
            static_cast<framework::proto::VarType::Type>(
                BOOST_GET_CONST(int, attr)));
        kernel_ctx->EmplaceBackAttr(data_type);
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<int64_t>))) {
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int>))) {
          // Emplace Back Attr according to the type of Pten_Kernel args.
          const auto& vector_int_attr = BOOST_GET_CONST(std::vector<int>, attr);
          const std::vector<int64_t> vector_int64_attr(vector_int_attr.begin(),
                                                       vector_int_attr.end());
          kernel_ctx->EmplaceBackAttr(vector_int64_attr);
        }
        // TODO(YuanRisheng) Need support vector<int64_t> attr
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported cast op attribute `%s` when construct "
            "KernelContext in dygraph.",
            attr_names[i]));
      }
    }
  }
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

  DygraphInferShapeContext<VarType> infer_shape_ctx(
      &ins, &outs, &attrs, &default_attrs, op.Type(), &kernel_type);
  op.Info().infer_shape_(&infer_shape_ctx);

  func(DygraphExecutionContext<VarType>(op, scope, *dev_ctx, ctx, ins, outs,
                                        attrs, default_attrs));

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
  DygraphInferShapeContext<VarType> infer_shape_ctx(
      &ins, &outs, &attrs, &default_attrs, op.Type(), &kernel_type);
  op.Info().infer_shape_(&infer_shape_ctx);

  PreparePtenData<VarType>(pt_kernel, pt_kernel_signature, ins);

  pten::KernelContext pt_kernel_context;
  BuildDygraphPtenKernelContext<VarType>(pt_kernel_signature, pt_kernel, ins,
                                         outs, attrs, default_attrs, dev_ctx,
                                         &pt_kernel_context);

  pt_kernel(&pt_kernel_context);

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

}  // namespace imperative
}  // namespace paddle
