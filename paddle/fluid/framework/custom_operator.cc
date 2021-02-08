/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/custom_operator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/tensor.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/c/c_api.h"
#include "paddle/fluid/framework/custom_tensor_utils.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_meta_info_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

namespace detail {

// dynamic lib load func
template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

inline bool IsGradVar(const std::string& var_name) {
  std::string suffix = kGradVarSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline std::string NoGrad(const std::string& var_name) {
  std::string suffix = kGradVarSuffix;
  return var_name.substr(0, var_name.size() - kGradVarSuffixSize);
}

inline bool IsMemberOf(const std::vector<std::string>& vec,
                       const std::string& name) {
  return std::find(vec.cbegin(), vec.cend(), name) != vec.cend();
}

}  // namespace detail

////////////////// Kernel Define ////////////////////

// custom op kernel call function define
static void RunKernelFunc(const framework::ExecutionContext& ctx,
                          const paddle::KernelFunc& func,
                          const std::vector<std::string>& inputs,
                          const std::vector<std::string>& outputs) {
  VLOG(1) << "Custom Operator: Start run KernelFunc.";
  std::vector<paddle::Tensor> custom_ins;
  for (auto& in_name : inputs) {
    VLOG(1) << "Custom Operator: input name - " << in_name;
    auto* x = ctx.Input<Tensor>(in_name);
    PADDLE_ENFORCE_NOT_NULL(x, platform::errors::NotFound(
                                   "Input tensor (%s) is nullptr.", in_name));
    PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "Input tensor (%s) is not initialized."));
    auto custom_in = paddle::Tensor(
        CustomTensorUtils::ConvertInnerPlaceToEnumPlace(x->place()));
    CustomTensorUtils::ShareDataFrom(static_cast<const void*>(x), custom_in);
    custom_ins.emplace_back(custom_in);
  }

  std::vector<boost::any> attrs;

  VLOG(1) << "Run ComputeFunc.";
  auto outs = func(custom_ins, attrs);

  VLOG(1) << "Custom Operator: Share outputs into ExecutionContext.";
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto* true_out = ctx.Output<Tensor>(outputs[i]);
    CustomTensorUtils::ShareDataTo(outs.at(i), true_out);
  }
}

//////////////////// Operator Define /////////////////

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  // Dummy infershape
  // Because it is a pure virtual function, it must be implemented
  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(1) << "Custom Operator: Dummy infer shape of custom operator.";
  }

  /**
   * NOTE: [Skip the Kernel Selection]
   * Custom Op only registers one Op kernel on each device, so that the
   * data type selection and promotion that depends on GetExpectedKernelType,
   * as well as the adaptation of various other special situations,
   * need users to implement, to avoid users needs to implement
   * GetExpectedKernelType function when expanding other cases.
   * The RAW type is used here as the data type, indicating that
   * it can only be determined at runtime.
   */
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(proto::VarType::RAW, ctx.GetPlace());
  }

  /**
   * NOTE: [Skip Input Variable Cast for DataType]
   * Because the kernel data type is RAW, we should skip the cast for
   * data type difference when PrepareData.
   */
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const OpKernelType& expected_kernel_type) {
    return OpKernelType(expected_kernel_type.data_type_,
                        expected_kernel_type.place_, tensor.layout());
  }
};

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const std::vector<std::string>& attrs)
      : inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  void Make() override {
    for (auto& in_name : inputs_) {
      AddInput(in_name, "The input " + in_name + "of Custom operator.");
    }
    for (auto& out_name : outputs_) {
      AddOutput(out_name, "The output " + out_name + "of Custom Operator.");
    }
    // TODO(chenweihang): support attrs in later PR
    AddComment(R"DOC(
Custom Operator.

According to the Tensor operation function implemented by the user 
independently of the framework, it is encapsulated into a framework 
operator to adapt to various execution scenarios such as dynamic graph, 
mode static graph mode, and inference mode.

)DOC");
  }

 private:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
};

template <typename T>
class CustomGradOpMaker;

template <>
class CustomGradOpMaker<OpDesc> : public SingleGradOpMaker<OpDesc> {
 public:
  explicit CustomGradOpMaker(
      const OpDesc& fwd_op, const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block, const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs)
      : SingleGradOpMaker<OpDesc>(fwd_op, no_grad_set, grad_to_var, grad_block),
        name_(name),
        inputs_(inputs),
        outputs_(outputs) {}

 protected:
  void Apply(GradOpPtr<OpDesc> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(1) << "Custom Operator: GradOpDescMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
      }
    }
    for (auto& out_name : outputs_) {
      VLOG(1) << "Custom Operator: GradOpDescMaker - output: " << out_name;
      grad_op->SetOutput(out_name, this->InputGrad(detail::NoGrad(out_name)));
    }
    // TODO(chenweihang): support attrs in later PR
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

template <>
class CustomGradOpMaker<imperative::OpBase>
    : public SingleGradOpMaker<imperative::OpBase> {
 public:
  explicit CustomGradOpMaker(
      const std::string& type,
      const imperative::NameVarBaseMap& var_base_map_in,
      const imperative::NameVarBaseMap& var_base_map_out,
      const AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map,
      const std::string& name, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs)
      : SingleGradOpMaker<imperative::OpBase>(
            type, var_base_map_in, var_base_map_out, attrs, inplace_map),
        name_(name),
        inputs_(inputs),
        outputs_(outputs) {}

 protected:
  // TODO(chenweihang): The code is duplicated with the previous one, because
  // ere OpMaker's Input, Output and other methods are protected. Putting the
  // function implementation outside the class will cause the method to be
  // uncallable,
  // so it is still implemented in the class for the time being.
  void Apply(GradOpPtr<imperative::OpBase> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(1) << "Custom Operator: GradOpBaseMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
      }
    }
    for (auto& out_name : outputs_) {
      VLOG(1) << "Custom Operator: GradOpBaseMaker - output: " << out_name;
      grad_op->SetOutput(out_name, this->InputGrad(detail::NoGrad(out_name)));
    }
    // TODO(chenweihang): support attrs in later PR
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

//////////// Operator and Kernel Register //////////////

void RegisterOperatorKernelWithPlace(const std::string& name,
                                     const paddle::KernelFunc& kernel_func,
                                     const proto::VarType::Type type,
                                     const PlaceType& place,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs) {
  OpKernelType key(type,
                   CustomTensorUtils::ConvertEnumPlaceToInnerPlace(place));
  VLOG(1) << "Custom Operator: op kernel key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [kernel_func, inputs, outputs](const framework::ExecutionContext& ctx) {
        VLOG(1) << "Custom Operator: run custom kernel func in lambda.";
        RunKernelFunc(ctx, kernel_func, inputs, outputs);
      };
}

void RegisterOperatorKernel(const std::string& name,
                            const paddle::KernelFunc& kernel_func,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs) {
  VLOG(1) << "Custom Operator: op name in kernel: " << name;
  // NOTE [ Dummy Op Kernel Key ]
  // TODO(chenweihang): Because execute engine need get device context based
  // op_kernel_key.place_, so we should register kernel for each
  // device. But this is not entirely correct, if user only give a cpu kernel,
  // but call api in gpu device, it will cause error.
  RegisterOperatorKernelWithPlace(name, kernel_func, proto::VarType::RAW,
                                  PlaceType::kCPU, inputs, outputs);
  RegisterOperatorKernelWithPlace(name, kernel_func, proto::VarType::RAW,
                                  PlaceType::kGPU, inputs, outputs);
}

void RegisterOperatorWithMetaInfo(
    const std::vector<OpMetaInfo>& op_meta_infos) {
  /* Op register */
  OpInfo info;

  auto& base_op_meta = op_meta_infos.front();

  auto op_name = OpMetaInfoHelper::GetOpName(base_op_meta);
  auto& op_inputs = OpMetaInfoHelper::GetInputs(base_op_meta);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(base_op_meta);
  auto& op_attrs = OpMetaInfoHelper::GetAttrs(base_op_meta);
  auto& kernel_fn = OpMetaInfoHelper::GetKernelFn(base_op_meta);
  auto& infer_shape_func = OpMetaInfoHelper::GetInferShapeFn(base_op_meta);
  auto& infer_dtype_func = OpMetaInfoHelper::GetInferDtypeFn(base_op_meta);

  VLOG(1) << "Custom Operator: forward, op name: " << op_name;
  VLOG(1) << "Custom Operator: forward, op inputs: "
          << string::join_strings(op_inputs, ',');
  VLOG(1) << "Custom Operator: forward, op outputs: "
          << string::join_strings(op_outputs, ',');

  // Op
  info.creator_ = [](const std::string& op_name, const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs) {
    return new CustomOperator(op_name, inputs, outputs, attrs);
  };

  // OpMaker
  info.proto_ = new proto::OpProto;
  info.proto_->set_type(op_name);

  info.checker_ = new OpAttrChecker();
  CustomOpMaker custom_maker(op_inputs, op_outputs, op_attrs);
  custom_maker(info.proto_, info.checker_);
  PADDLE_ENFORCE_EQ(
      info.proto_->IsInitialized(), true,
      platform::errors::PreconditionNotMet(
          "Fail to initialize %s's OpProto, because %s is not initialized.",
          op_name, info.proto_->InitializationErrorString()));

  // InferShape
  PADDLE_ENFORCE_NOT_NULL(
      infer_shape_func,
      platform::errors::PreconditionNotMet(
          "InferShapeFn is nullptr. Need to set the InferShapeFn of custom "
          "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));
  info.infer_shape_ = [op_inputs, op_outputs,
                       infer_shape_func](InferShapeContext* ctx) {
    std::vector<std::vector<int64_t>> input_shapes;

    VLOG(1) << "Custom Operator: InferShape - get input ddim.";
    for (auto& in_name : op_inputs) {
      OP_INOUT_CHECK(ctx->HasInput(in_name), "Input", in_name, "Custom");
      auto ddim = ctx->GetInputDim(in_name);
      input_shapes.emplace_back(framework::vectorize(ddim));
    }

    VLOG(1) << "Custom Operator: InferShape - calc output ddim.";
    auto output_shapes = infer_shape_func(input_shapes);

    VLOG(1) << "Custom Operator: InferShape - set output ddim.";
    for (size_t i = 0; i < op_outputs.size(); ++i) {
      ctx->SetOutputDim(op_outputs[i], framework::make_ddim(output_shapes[i]));
    }
  };

  // Infer Dtype
  PADDLE_ENFORCE_NOT_NULL(
      infer_dtype_func,
      platform::errors::PreconditionNotMet(
          "InferDtypeFn is nullptr. Need to set the InferDtypeFn of custom "
          "operator by .SetInferDtypeFn(PD_INFER_DTYPE(...))"));
  info.infer_var_type_ = [op_inputs, op_outputs,
                          infer_dtype_func](InferVarTypeContext* ctx) {
    std::vector<DataType> input_dtypes;

    VLOG(1) << "Custom Operator: InferDtype - get input dtype.";
    for (auto& in_name : op_inputs) {
      auto dtype = ctx->GetInputDataType(in_name);
      input_dtypes.emplace_back(
          CustomTensorUtils::ConvertInnerDTypeToEnumDType(dtype));
    }

    VLOG(1) << "Custom Operator: InferDtype - infer output dtype.";
    auto output_dtypes = infer_dtype_func(input_dtypes);

    VLOG(1) << "Custom Operator: InferDtype - set output dtype.";
    for (size_t i = 0; i < op_outputs.size(); ++i) {
      ctx->SetOutputDataType(
          op_outputs[i],
          CustomTensorUtils::ConvertEnumDTypeToInnerDType(output_dtypes[i]));
    }
  };

  // Kernel func
  RegisterOperatorKernel(op_name, kernel_fn, op_inputs, op_outputs);

  // If grad op or double grad op exists
  std::string cur_op_name = op_name;
  for (size_t i = 1; i < op_meta_infos.size(); ++i) {
    auto& cur_grad_op = op_meta_infos[i];

    auto& grad_op_name = OpMetaInfoHelper::GetOpName(cur_grad_op);
    auto& grad_op_inputs = OpMetaInfoHelper::GetInputs(cur_grad_op);
    auto& grad_op_outputs = OpMetaInfoHelper::GetOutputs(cur_grad_op);
    auto& grad_kernel_fn = OpMetaInfoHelper::GetKernelFn(cur_grad_op);

    VLOG(1) << "Custom Operator: backward, op name: " << grad_op_name;
    VLOG(1) << "Custom Operator: backward, op inputs: "
            << string::join_strings(grad_op_inputs, ',');
    VLOG(1) << "Custom Operator: backward, op outputs: "
            << string::join_strings(grad_op_outputs, ',');

    // GradOpDescMaker
    info.grad_op_maker_ = [grad_op_name, grad_op_inputs, grad_op_outputs](
        const OpDesc& fwd_op,
        const std::unordered_set<std::string>& no_grad_set,
        std::unordered_map<std::string, std::string>* grad_to_var,
        const std::vector<BlockDesc*>& grad_block) {
      CustomGradOpMaker<paddle::framework::OpDesc> maker(
          fwd_op, no_grad_set, grad_to_var, grad_block, grad_op_name,
          grad_op_inputs, grad_op_outputs);
      return maker();
    };

    // GradOpBaseMaker
    info.dygraph_grad_op_maker_ = [grad_op_name, grad_op_inputs,
                                   grad_op_outputs](
        const std::string& type,
        const imperative::NameVarBaseMap& var_base_map_in,
        const imperative::NameVarBaseMap& var_base_map_out,
        const framework::AttributeMap& attrs,
        const std::map<std::string, std::string>& inplace_map) {
      CustomGradOpMaker<paddle::imperative::OpBase> maker(
          type, var_base_map_in, var_base_map_out, attrs, inplace_map,
          grad_op_name, grad_op_inputs, grad_op_outputs);
      return maker();
    };

    /* Grad op register */
    OpInfo grad_info;

    // Grad Op
    grad_info.creator_ = [](
        const std::string& type, const VariableNameMap& inputs,
        const VariableNameMap& outputs, const AttributeMap& attrs) {
      return new CustomOperator(type, inputs, outputs, attrs);
    };

    // Grad InferShape (gradient's shape is same with forward input default)
    grad_info.infer_shape_ = [grad_op_outputs](InferShapeContext* ctx) {
      for (auto& out_name : grad_op_outputs) {
        ctx->ShareDim(detail::NoGrad(out_name), out_name);
      }
    };

    // Kernel func
    RegisterOperatorKernel(grad_op_name, grad_kernel_fn, grad_op_inputs,
                           grad_op_outputs);

    // update current info
    OpInfoMap::Instance().Insert(cur_op_name, info);
    cur_op_name = grad_op_name;
    info = grad_info;
  }
  // insert last info
  OpInfoMap::Instance().Insert(cur_op_name, info);
}

void RegisterOperatorWithMetaInfoMap(
    const paddle::OpMetaInfoMap& op_meta_info_map) {
  auto& meta_info_map = op_meta_info_map.GetMap();

  PADDLE_ENFORCE_EQ(meta_info_map.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "No custom operator that needs to be registered."));
  VLOG(1) << "Custom Operator: size of op meta info map - "
          << meta_info_map.size();
  // pair: {op_type, OpMetaInfo}
  for (auto& pair : meta_info_map) {
    VLOG(1) << "Custom Operator: pair first -> op name: " << pair.first;
    RegisterOperatorWithMetaInfo(pair.second);
  }
}

////////////////////// User APIs ///////////////////////

// load op api
void LoadOpMetaInfoAndRegisterOp(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpMetaInfoMap& get_op_meta_info_map_t();
  auto* get_op_meta_info_map =
      detail::DynLoad<get_op_meta_info_map_t>(handle, "PD_GetOpMetaInfoMap");
  auto& op_meta_info_map = get_op_meta_info_map();

  RegisterOperatorWithMetaInfoMap(op_meta_info_map);
}

}  // namespace framework
}  // namespace paddle
