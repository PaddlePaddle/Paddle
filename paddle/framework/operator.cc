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
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>

#include "paddle/framework/data_transform.h"
#include "paddle/framework/device_data_transform.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/shape_inference.h"
#include "paddle/framework/var_type.h"

DEFINE_bool(op_sync, false,
            "Default cuda is asynchronous device, set to True will"
            "force op run in synchronous mode.");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

void UseCPU() {
  kKernelPriority.clear();
  /*Plain CPU*/
  auto pair0 = std::make_tuple(platform::CPUPlace(), LibraryType::kPlain);
  kKernelPriority.insert(kKernelPriority.begin(), pair0);
}

void UseMKLDNN() {
  UseCPU();
#if PADDLE_WITH_MKLML
  {
    /*MKLDNN Kernel*/
    auto pair0 = std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN);
    kKernelPriority.insert(kKernelPriority.begin(), pair0);
  }
#endif
}

void UseCUDA() {
  UseMKLDNN();
#if PADDLE_WITH_CUDA
  /*Plain GPU*/
  auto pair0 = std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain);
  kKernelPriority.insert(kKernelPriority.begin(), pair0);
#endif
}

void UseCUDNN() {
  UseCUDA();
#if PADDLE_WITH_CUDA
  if (platform::dynload::HasCUDNN()) {
    /*CUDNN Kernel*/
    auto pair0 = std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN);
    kKernelPriority.insert(kKernelPriority.begin(), pair0);
  }
#endif
}

void UseALL() {
  UseCPU();
  UseMKLDNN();
  UseCUDA();
  UseCUDNN();
}

static DDim GetDims(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  } else if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().dims();
  } else if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().GetCompleteDims();
  } else {
    return DDim({-1});
  }
}

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

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (scope) {
        ss << "(" << GetDims(*scope, input.second[i]) << ")";
      }
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
      if (scope) {
        ss << "(" << GetDims(*scope, output.second[i]) << ")";
      }
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

static bool VarIsTensor(const Variable* var) {
  return var->IsType<LoDTensor>() || var->IsType<SelectedRows>();
}

static const Tensor* GetTensorFromVar(const Variable* var) {
  const Tensor* t = nullptr;
  if (var->IsType<LoDTensor>()) {
    t = &(var->Get<LoDTensor>());
  } else if (var->IsType<SelectedRows>()) {
    t = &(var->Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
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
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
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

    // TODO(dzhwinter) : reuse ShareLoD in most operators.
    // Need to call ShareLayout explicitly in sequence related ops.
    // Shall we have a better method to shared info between in/out Tensor?
    out_tensor->set_layout(in_tensor.layout());
  }

  void ShareLayout(const std::string& in, const std::string& out, size_t i = 0,
                   size_t j = 0) const {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    Variable* in_var = scope_.FindVar(Inputs(in)[i]);
    Variable* out_var = scope_.FindVar(Outputs(out)[j]);
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_layout(in_tensor.layout());
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
      PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
                   name, var->Type().name());
    }
  }

  void SetDim(const std::string& name, const DDim& dim) override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
                   name, var->Type().name());
    }
  }

  proto::VarDesc::VarType GetVarType(const std::string& name) const override {
    auto* var = scope_.FindVar(name);
    return ToVarType(var->Type());
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

void OperatorWithKernel::Run(const Scope& scope,
                             const platform::Place& place) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope);
  this->InferShape(&infer_shape_ctx);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  ExecutionContext ctx(*this, scope, *dev_ctx);
  auto expected_kernel_key = this->GetExpectedKernelType(ctx);

  OpKernelMap& kernels = kernels_iter->second;

  for (auto& candidate : kKernelPriority) {
    auto candidate_key =
        OpKernelType(expected_kernel_key.data_type_, std::get<0>(candidate),
                     expected_kernel_key.data_layout_, std::get<1>(candidate));

    if ((candidate_key == expected_kernel_key) ||
        (kernels.count(candidate_key))) {
      expected_kernel_key = candidate_key;
      break;
    }
  }

  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  Scope& new_scope = scope.NewScope();

  for (auto& var_name_item : this->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto* var = scope.FindVar(var_name);
      if (var && VarIsTensor(var)) {
        auto* tensor_in = GetTensorFromVar(var);
        if (tensor_in->IsInitialized()) {
          auto kernel_type_for_var = this->GetKernelTypeForVar(
              var_name_item.first, *tensor_in, expected_kernel_key);
          if (kernel_type_for_var != expected_kernel_key) {
            auto out_var_names = OutputVars(true);
            if (std::find(out_var_names.begin(), out_var_names.end(),
                          var_name) != out_var_names.end()) {
              PADDLE_THROW(
                  "var %s is both input and output, "
                  "does not support transform",
                  var_name);
            }
            VLOG(3) << "need to do transform for var " << var_name;
            auto* trans_var = new_scope.Var(var_name);
            auto* out = DataTransform(expected_kernel_key, kernel_type_for_var,
                                      *tensor_in);
            CopyVariableWithTensor(*var, *out, *trans_var);
          }
        }
      }
    }
  }

  auto kernel_iter = kernels.find(expected_kernel_key);

  auto* new_dev_ctx = pool.Get(expected_kernel_key.place_);
  kernel_iter->second->Compute(
      ExecutionContext(*this, new_scope, *new_dev_ctx));

  /*For profiling/benchmark only*/
  if (FLAGS_op_sync) {
    new_dev_ctx->Wait();
  }
}

proto::DataType OperatorWithKernel::IndicateDataType(
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
  return static_cast<proto::DataType>(data_type);
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(expected_kernel_type.data_type_, tensor.place());
}

}  // namespace framework
}  // namespace paddle
