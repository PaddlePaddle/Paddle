/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include <algorithm>
#include <atomic>
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/shape_inference.h"
#include "paddle/framework/var_type.h"

namespace paddle {
namespace framework {

template <>
Eigen::DefaultDevice& ExecutionContext::GetEigenDevice<
    platform::CPUPlace, Eigen::DefaultDevice>() const {
  return *device_context_.GetEigenDevice<platform::CPUPlace>();
}

#ifdef PADDLE_WITH_CUDA
template <>
Eigen::GpuDevice&
ExecutionContext::GetEigenDevice<platform::GPUPlace, Eigen::GpuDevice>() const {
  return *device_context_.GetEigenDevice<platform::GPUPlace>();
}
#endif

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

void OperatorBase::Rename(const std::string& old_name,
                          const std::string& new_name) {
  for (auto& input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }
  for (auto& output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = OpInfoMap::Instance().Get(Type());

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  auto& info_map = OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) return;

  for (auto& in : op_info->Proto().inputs()) {
    PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                   "Type %s's input %s is not set", Type(), in.name());
  }

  for (auto& out : op_info->Proto().outputs()) {
    PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                   "Type %s's output %s is not set", Type(), out.name());
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static const Tensor* GetTensorFromVar(const Variable* var) {
  const Tensor* t = nullptr;
  if (var->IsType<LoDTensor>()) {
    t = &(var->Get<LoDTensor>());
  } else if (var->IsType<SelectedRows>()) {
    t = &(var->Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
  }
  return t;
}

static Tensor* GetMutableTensorFromVar(Variable* var) {
  Tensor* t = nullptr;
  if (var->IsType<LoDTensor>()) {
    t = var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    t = var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
  }
  return t;
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  auto* var = InputVar(name);
  return var == nullptr ? nullptr : GetTensorFromVar(var);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto names = op().Inputs(name);
  std::vector<const Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr : GetTensorFromVar(var);
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  auto var = OutputVar(name);
  return var == nullptr ? nullptr : GetMutableTensorFromVar(var);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto names = op().Outputs(name);
  std::vector<Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr
                                         : GetMutableTensorFromVar(var);
                 });
  return res;
}

std::ostream& operator<<(std::ostream& os, const OpKernelType& kernel_key) {
  os << "place[" << kernel_key.place_ << "]:data_type[" << kernel_key.data_type_
     << "]";
  return os;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  bool HasInput(const std::string& name) const override {
    auto& ins = Inputs(name);
    size_t length = ins.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL, "Input %s should have more than one inputs",
                      name);
    auto ipt = ins[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    auto& outs = Outputs(name);
    size_t length = outs.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL, "Output %s should have more than one inputs",
                      name);
    auto ipt = outs[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    auto inputs = op_.Inputs(name);
    if (inputs.empty()) {
      return false;
    }
    for (auto& input : inputs) {
      if (scope_.FindVar(input) == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    auto outputs = op_.Outputs(name);
    if (outputs.empty()) {
      return false;
    }
    for (auto& output : outputs) {
      if (scope_.FindVar(output) == nullptr) {
        return false;
      }
    }
    return true;
  }

  DDim GetInputDim(const std::string& name) const override {
    return GetDim(op_.Input(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    SetDim(op_.Output(name), dim);
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    Variable* in_var = scope_.FindVar(Inputs(in)[i]);
    Variable* out_var = scope_.FindVar(Outputs(out)[j]);
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
  }

  bool IsRuntime() const override { return true; }

 protected:
  DDim GetDim(const std::string& name) const override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
    }
  }

  void SetDim(const std::string& name, const DDim& dim) override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
    }
  }

  VarDesc::VarType GetVarType(const std::string& name) const override {
    auto* var = scope_.FindVar(name);
    return ToVarType(var->Type());
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

void OperatorWithKernel::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  if (VLOG_IS_ON(1)) {
    auto inputs = this->InputVars();
    auto outputs = this->OutputVars(true);
    std::ostringstream sout;
    sout << "Run operator " << this->Type() << " From [";
    std::ostream_iterator<std::string> out_it(sout, ",");
    std::copy(inputs.begin(), inputs.end(), out_it);
    sout << "] to [";
    std::copy(outputs.begin(), outputs.end(), out_it);
    sout << "]";
    VLOG(1) << sout.str();
  }

  RuntimeInferShapeContext infer_shape_ctx(*this, scope);
  this->InferShape(&infer_shape_ctx);

  ExecutionContext ctx(*this, scope, dev_ctx);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  // check if op[type] have kernel for kernel_key
  OpKernelMap& kernels = kernels_iter->second;
  auto kernel_key = GetKernelType(ctx);
  auto kernel_iter = kernels.find(kernel_key);

  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("The operator %s does not support %s", type_, kernel_key);
  }

  kernel_iter->second->Compute(ctx);

  // throws errors if have.
  dev_ctx.Finish();
}
OpKernelType OperatorWithKernel::GetKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.device_context());
}
DataType OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  auto& scope = ctx.scope();
  int data_type = -1;
  for (auto& input : this->inputs_) {
    for (auto& ipt_name : input.second) {
      auto* var = scope.FindVar(ipt_name);
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          int tmp = static_cast<int>(ToDataType(t->type()));
          PADDLE_ENFORCE(tmp == data_type || data_type == -1,
                         "DataType of Paddle Op %s must be the same.", Type());
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
  return static_cast<DataType>(data_type);
}

}  // namespace framework
}  // namespace paddle
