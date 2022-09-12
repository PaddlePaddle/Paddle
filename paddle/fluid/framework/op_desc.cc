/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_desc.h"

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/utils/blank.h"

namespace paddle {
namespace framework {

class CompileTimeInferShapeContext : public InferShapeContext {
 public:
  CompileTimeInferShapeContext(const OpDesc &op, const BlockDesc &block);

  bool HasInput(const std::string &name) const override;

  bool HasOutput(const std::string &name) const override;

  bool HasAttr(const std::string &name) const override;

  bool HasInputs(const std::string &name) const override;

  bool HasOutputs(const std::string &name,
                  bool allow_null = false) const override;

  AttrReader Attrs() const override;

  std::vector<std::string> Inputs(const std::string &name) const override;

  std::vector<std::string> Outputs(const std::string &name) const override;

  std::string GetInputNameByIdx(size_t idx) const override {
    auto &op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(idx,
                      op_proto->inputs().size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          op_.Type(),
                          idx,
                          op_proto->inputs().size()));
    return op_proto->inputs()[idx].name();
  }

  std::string GetOutputNameByIdx(size_t idx) const override {
    auto &op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(
        idx,
        op_proto->outputs().size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            op_.Type(),
            idx,
            op_proto->outputs().size()));
    return op_proto->outputs()[idx].name();
  }

  void ShareDim(const std::string &in,
                const std::string &out,
                size_t i = 0,
                size_t j = 0) override {
    PADDLE_ENFORCE_LT(i,
                      Inputs(in).size(),
                      platform::errors::InvalidArgument(
                          "The input variable index is out of range, expected "
                          "index less than %d, but received index is %d.",
                          Inputs(in).size(),
                          i));
    PADDLE_ENFORCE_LT(j,
                      Outputs(out).size(),
                      platform::errors::InvalidArgument(
                          "The output variable index is out of range, expected "
                          "index less than %d, but received index is %d.",
                          Outputs(out).size(),
                          j));

    std::string input_n = Inputs(in)[i];
    std::string output_n = Outputs(out)[j];

    PADDLE_ENFORCE_NE(input_n,
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The input variable %s[%d] is empty.", in, i));
    PADDLE_ENFORCE_NE(output_n,
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The output variable %s[%d] is empty.", out, j));

    auto *in_var = block_.FindVarRecursive(input_n);
    auto *out_var = block_.FindVarRecursive(output_n);

    PADDLE_ENFORCE_EQ(
        in_var->GetType(),
        out_var->GetType(),
        platform::errors::InvalidArgument(
            "The type of input %s and output %s do not match. The input type "
            "is %s, output type is %s.",
            input_n,
            output_n,
            DataTypeToString(in_var->GetType()),
            DataTypeToString(out_var->GetType())));

    SetDim(output_n, GetDim(input_n));
  }

  void ShareAllLoD(const std::string &in,
                   const std::string &out) const override {
    auto &in_var_names = op_.Input(in);
    auto &out_var_names = op_.Output(out);

    PADDLE_ENFORCE_EQ(
        in_var_names.size(),
        out_var_names.size(),
        platform::errors::PreconditionNotMet(
            "Op [%s]:  Input var number should be equal with output var number",
            op_.Type()));

    for (size_t i = 0; i < in_var_names.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) {
        continue;
      }

      auto *in_var = block_.FindVarRecursive(in_var_names[i]);
      auto *out_var = block_.FindVarRecursive(out_var_names[i]);
      if (in_var->GetType() != proto::VarType::LOD_TENSOR &&
          in_var->GetType() != proto::VarType::LOD_TENSOR_ARRAY) {
        VLOG(3) << "input " << in << " is not LoDTensor or LoDTensorArray.";
        return;
      }
      out_var->SetLoDLevel(in_var->GetLoDLevel());
    }
  }

  void ShareLoD(const std::string &in,
                const std::string &out,
                size_t i = 0,
                size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i,
                      Inputs(in).size(),
                      platform::errors::InvalidArgument(
                          "The input variable index is out of range, expected "
                          "index less than %d, but received index is %d.",
                          Inputs(in).size(),
                          i));
    PADDLE_ENFORCE_LT(j,
                      Outputs(out).size(),
                      platform::errors::InvalidArgument(
                          "The output variable index is out of range, expected "
                          "index less than %d, but received index is %d.",
                          Outputs(out).size(),
                          j));
    PADDLE_ENFORCE_NE(Inputs(in)[i],
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The input variable %s[%d] is empty.", in, i));
    PADDLE_ENFORCE_NE(Outputs(out)[j],
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The output variable %s[%d] is empty.", out, j));
    auto *in_var = block_.FindVarRecursive(Inputs(in)[i]);
    auto *out_var = block_.FindVarRecursive(Outputs(out)[j]);
    if (in_var->GetType() != proto::VarType::LOD_TENSOR &&
        in_var->GetType() != proto::VarType::LOD_TENSOR_ARRAY) {
      VLOG(3) << "input " << in << " is not LoDTensor or LoDTensorArray.";
      return;
    }
    out_var->SetLoDLevel(in_var->GetLoDLevel());
  }

  int32_t GetLoDLevel(const std::string &in, size_t i = 0) const override {
    PADDLE_ENFORCE_LT(i,
                      Inputs(in).size(),
                      platform::errors::InvalidArgument(
                          "The input variable index is out of range, input "
                          "variable %s of operator %s only has %d elements.",
                          in,
                          op_.Type(),
                          Inputs(in).size()));
    PADDLE_ENFORCE_NE(Inputs(in)[i],
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The input variable %s[%d] of operator %s is empty.",
                          in,
                          i,
                          op_.Type()));
    auto *in_var = block_.FindVarRecursive(Inputs(in)[i]);
    PADDLE_ENFORCE_NOT_NULL(
        in_var,
        platform::errors::NotFound(
            "The input variable %s[%d] of operator %s is not found.",
            in,
            i,
            op_.Type()));
    return in_var->GetLoDLevel();
  }

  void SetLoDLevel(const std::string &out,
                   int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_ENFORCE_LT(j,
                      Outputs(out).size(),
                      platform::errors::InvalidArgument(
                          "The output variable index is out of range, output "
                          "variable %s of operator %s only has %d elements.",
                          out,
                          op_.Type(),
                          Outputs(out).size()));
    PADDLE_ENFORCE_NE(Outputs(out)[j],
                      framework::kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "The output variable %s[%d] of operator %s is empty.",
                          out,
                          j,
                          op_.Type()));
    auto *out_var = block_.FindVarRecursive(Outputs(out)[j]);
    PADDLE_ENFORCE_NOT_NULL(
        out_var,
        platform::errors::NotFound(
            "The output variable %s[%d] of operator %s is not found.",
            out,
            j,
            op_.Type()));
    if (lod_level >= 0) {
      out_var->SetLoDLevel(lod_level);
    }
  }

  paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize>
  GetInputVarPtrs(const std::string &name) const override {
    const std::vector<std::string> arg_names = Inputs(name);
    paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize> res;
    res.reserve(arg_names.size());
    std::transform(arg_names.begin(),
                   arg_names.end(),
                   std::back_inserter(res),
                   [this](const std::string &name) {
                     return block_.FindVarRecursive(name);
                   });
    return res;
  }

  paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize>
  GetOutputVarPtrs(const std::string &name) const override {
    const std::vector<std::string> arg_names = Outputs(name);
    paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize> res;
    res.reserve(arg_names.size());
    std::transform(arg_names.begin(),
                   arg_names.end(),
                   std::back_inserter(res),
                   [this](const std::string &name) {
                     return block_.FindVarRecursive(name);
                   });
    return res;
  }

  DDim GetInputDim(const std::string &name) const override {
    const std::vector<std::string> &arg_names = Inputs(name);
    PADDLE_ENFORCE_EQ(arg_names.size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input(%s) should hold only one element, but now "
                          "it holds %d elements.",
                          name,
                          arg_names.size()));
    return this->GetDim(arg_names[0]);
  }

  std::vector<DDim> GetInputsDim(const std::string &name) const override {
    const std::vector<std::string> &arg_names = Inputs(name);
    return GetDims(arg_names);
  }

  bool IsRuntime() const override;

  bool IsRunMKLDNNKernel() const override;

  proto::VarType::Type GetInputVarType(const std::string &name) const override {
    return GetVarType(Inputs(name).at(0));
  }

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string &name) const override {
    return GetVarTypes(Inputs(name));
  }

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string &name) const override {
    return GetVarTypes(Outputs(name));
  }

  void SetOutputDim(const std::string &name, const DDim &dim) override {
    auto arg_names = Outputs(name);
    PADDLE_ENFORCE_EQ(arg_names.size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The iutput(%s) should hold only one element, but "
                          "now it holds %d elements.",
                          name,
                          arg_names.size()));
    SetDim(arg_names[0], dim);
  }

  void SetOutputsDim(const std::string &name,
                     const std::vector<DDim> &dims) override {
    auto names = Outputs(name);
    SetDims(names, dims);
  }

  const phi::ArgumentMappingFn *GetPhiArgumentMappingFn() const override {
    return phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_.Type());
  }

  const phi::KernelSignature *GetPhiDefaultKernelSignature() const override {
    return &phi::DefaultKernelSignatureMap::Instance().Get(op_.Type());
  }

 protected:
  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<std::string> &names) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(names.size());
    std::transform(
        names.begin(),
        names.end(),
        retv.begin(),
        std::bind(std::mem_fn(&CompileTimeInferShapeContext::GetVarType),
                  this,
                  std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(const std::string &name) const;

  DDim GetDim(const std::string &name) const {
    auto var = block_.FindVarRecursive(name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable %s is not found.", name));
    DDim res;
    try {
      auto shape = var->GetShape();
      res = shape.empty() ? phi::make_ddim({0UL}) : phi::make_ddim(shape);
    } catch (...) {
      VLOG(5) << "GetDim of variable " << name << " error";
      std::rethrow_exception(std::current_exception());
    }
    return res;
  }

  std::vector<DDim> GetDims(const std::vector<std::string> &names) const {
    std::vector<DDim> ret;
    ret.reserve(names.size());
    std::transform(
        names.begin(),
        names.end(),
        std::back_inserter(ret),
        [this](const std::string &name) { return this->GetDim(name); });
    return ret;
  }

  void SetDim(const std::string &name, const DDim &dim);

  void SetDims(const std::vector<std::string> &names,
               const std::vector<DDim> &dims) {
    size_t length = names.size();
    PADDLE_ENFORCE_EQ(length,
                      dims.size(),
                      platform::errors::InvalidArgument(
                          "The input variables number(%d) and input dimensions "
                          "number(%d) do not match.",
                          length,
                          dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (names[i] == framework::kEmptyVarName) {
        continue;
      }
      SetDim(names[i], dims[i]);
    }
  }

  std::vector<DDim> GetRepeatedDims(const std::string &name) const override;

  void SetRepeatedDims(const std::string &name,
                       const std::vector<DDim> &dims) override;

  const OpDesc &op_;
  const BlockDesc &block_;
};

static void InitRuntimeAttributeMapByOpExtraInfo(const std::string &op_type,
                                                 AttributeMap *runtime_attrs) {
  const auto &extra_attr_map =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(op_type);
  runtime_attrs->insert(extra_attr_map.begin(), extra_attr_map.end());
}

OpDesc::OpDesc(const std::string &type,
               const VariableNameMap &inputs,
               const VariableNameMap &outputs,
               const AttributeMap &attrs) {
  desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
  need_update_ = true;
  block_ = nullptr;
  InitRuntimeAttributeMapByOpExtraInfo(type, &runtime_attrs_);
}

OpDesc::OpDesc(const OpDesc &other) {
  CopyFrom(other);
  block_ = other.block_;
  need_update_ = true;
}

OpDesc::OpDesc(const OpDesc &other, BlockDesc *block) {
  CopyFrom(other);
  block_ = block;
  need_update_ = true;
  for (auto &iter : attrs_) {
    UpdateVarAttr(iter.first, iter.second);
  }
}

void OpDesc::CopyFrom(const OpDesc &op_desc) {
  desc_.set_type(op_desc.Type());
  inputs_ = op_desc.inputs_;
  outputs_ = op_desc.outputs_;
  attrs_ = op_desc.attrs_;
  runtime_attrs_ = op_desc.runtime_attrs_;
  // The record of original_id_ is only for auto parallel.
  original_id_ = op_desc.original_id_;
  if (op_desc.dist_attr_) {
    dist_attr_.reset(new OperatorDistAttr(*op_desc.dist_attr_));
  }
  need_update_ = true;
}

OpDesc::OpDesc(const proto::OpDesc &desc, BlockDesc *block)
    : desc_(desc), need_update_(false) {
  // restore inputs_
  int input_size = desc_.inputs_size();
  for (int i = 0; i < input_size; ++i) {
    const proto::OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore outputs_
  int output_size = desc_.outputs_size();
  for (int i = 0; i < output_size; ++i) {
    const proto::OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore attrs_
  InitRuntimeAttributeMapByOpExtraInfo(desc.type(), &runtime_attrs_);
  for (const proto::OpDesc::Attr &attr : desc_.attrs()) {
    const std::string &attr_name = attr.name();
    // The sub_block referred to by the BLOCK attr hasn't been added
    // to ProgramDesc class yet, we skip setting BLOCK/BLOCKS/VAR/VARS attr
    // here.
    auto attr_type = attr.type();
    if (attr_type != proto::AttrType::BLOCK &&
        attr_type != proto::AttrType::BLOCKS &&
        attr_type != proto::AttrType::VAR &&
        attr_type != proto::AttrType::VARS) {
      auto iter = runtime_attrs_.find(attr_name);
      if (iter == runtime_attrs_.end()) {
        attrs_[attr_name] = GetAttrValue(attr);
      } else {
        iter->second = GetAttrValue(attr);
      }
    }
  }
  this->block_ = block;
}

// Explicitly implement the assign operator, Since the added
// unique_ptr data member does not have the implicit assign operator.
OpDesc &OpDesc::operator=(const OpDesc &other) {
  CopyFrom(other);
  block_ = other.block_;
  need_update_ = true;
  return *this;
}

proto::OpDesc *OpDesc::Proto() {
  Flush();
  return &desc_;
}

const std::vector<std::string> &OpDesc::Input(const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      inputs_.end(),
      platform::errors::NotFound(
          "Input %s cannot be found in operator %s.", name, Type()));
  return it->second;
}

std::vector<std::string> OpDesc::Input(const std::string &name,
                                       bool with_attr_var) const {
  // Attribute with VarDesc type will consider as Input
  if (with_attr_var) {
    auto it = attrs_.find(name);
    if (it != attrs_.end() && HasAttrVar(it->second))
      return AttrVarNames(it->second);
  }
  return this->Input(name);
}

VariableNameMap OpDesc::Inputs(bool with_attr_var) const {
  if (!with_attr_var) {
    return inputs_;
  }
  VariableNameMap res = inputs_;
  for (auto &attr : FilterAttrVar(attrs_)) {
    res[attr.first] = AttrVarNames(attr.second);
  }
  return res;
}

std::vector<std::string> OpDesc::InputArgumentNames(bool with_attr_var) const {
  std::vector<std::string> retv;
  for (auto &ipt : this->Inputs(with_attr_var)) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetInput(const std::string &param_name,
                      const std::vector<std::string> &args) {
  need_update_ = true;
  inputs_[param_name] = args;
}

const std::vector<std::string> &OpDesc::Output(const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      outputs_.end(),
      platform::errors::NotFound(
          "Output %s cannot be found in operator %s.", name, Type()));
  return it->second;
}

bool OpDesc::HasOutput(const std::string &name) const {
  return outputs_.find(name) != outputs_.end();
}

std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->outputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetOutput(const std::string &param_name,
                       const std::vector<std::string> &args) {
  need_update_ = true;
  this->outputs_[param_name] = args;
}

void OpDesc::RemoveOutput(const std::string &name) {
  outputs_.erase(name);
  need_update_ = true;
}

void OpDesc::RemoveInput(const std::string &name) {
  inputs_.erase(name);
  need_update_ = true;
}

bool OpDesc::HasProtoAttr(const std::string &name) const {
  auto &op_info = OpInfoMap::Instance();
  if (op_info.Has(desc_.type())) {
    auto op_info_ptr = op_info.Get(desc_.type());
    if (op_info_ptr.HasOpProtoAndChecker()) {
      const proto::OpProto &proto = op_info_ptr.Proto();
      for (int i = 0; i != proto.attrs_size(); ++i) {
        const proto::OpProto::Attr &attr = proto.attrs(i);
        if (attr.name() == name) {
          return true;
        }
      }
    }
  }
  return false;
}

proto::AttrType OpDesc::GetAttrType(const std::string &name,
                                    bool with_attr_var) const {
  auto attr = this->GetAttr(name, with_attr_var);
  return static_cast<proto::AttrType>(attr.index() - 1);
}

std::vector<std::string> OpDesc::AttrNames(bool with_attr_var) const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    if (!with_attr_var && HasAttrVar(attr.second)) continue;
    retv.push_back(attr.first);
  }
  return retv;
}

bool OpDesc::HasAttr(const std::string &name, bool with_attr_var) const {
  auto iter = attrs_.find(name);
  bool is_found = true;
  if (iter == attrs_.end()) {
    iter = runtime_attrs_.find(name);
    if (iter == runtime_attrs_.end()) {
      is_found = false;
    }
  }
  if (with_attr_var) {
    return is_found;
  }
  return is_found && !HasAttrVar(iter->second);
}

void OpDesc::RemoveAttr(const std::string &name) {
  attrs_.erase(name);
  runtime_attrs_.erase(name);
  need_update_ = true;
}

void OpDesc::SetAttr(const std::string &name, const Attribute &v) {
  AttributeMap *attrs_ptr = &(this->attrs_);

  bool is_runtime_attr = false;

  const auto &extra_attr_map =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsMap(Type());
  auto extra_attr_iter = extra_attr_map.find(name);
  if (extra_attr_iter != extra_attr_map.end()) {
    is_runtime_attr = true;
    attrs_ptr = &(this->runtime_attrs_);
  }
  // NOTICE(minqiyang): pybind11 will take the empty list in python as
  // the std::vector<int> type in C++; so we have to change the attr's type
  // here if we meet this issue
  proto::AttrType attr_type = static_cast<proto::AttrType>(v.index() - 1);
  if (attr_type == proto::AttrType::INTS &&
      PADDLE_GET_CONST(std::vector<int>, v).size() == 0u) {
    // Find current attr via attr name and set the correct attribute value
    auto attr_type =
        is_runtime_attr
            ? static_cast<proto::AttrType>(extra_attr_iter->second.index() - 1)
            : GetProtoAttr(name).type();
    switch (attr_type) {
      case proto::AttrType::BOOLEANS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to BOOLEANS";
        attrs_ptr->operator[](name) = std::vector<bool>();
        break;
      }
      case proto::AttrType::INTS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to INTS";
        attrs_ptr->operator[](name) = std::vector<int>();
        break;
      }
      case proto::AttrType::LONGS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from LONGS to LONGS";
        attrs_ptr->operator[](name) = std::vector<int64_t>();
        break;
      }
      case proto::AttrType::FLOATS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to FLOATS";
        attrs_ptr->operator[](name) = std::vector<float>();
        break;
      }
      case proto::AttrType::FLOAT64S: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to FLOAT64S";
        this->attrs_[name] = std::vector<double>();
        break;
      }
      case proto::AttrType::STRINGS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to STRINGS";
        attrs_ptr->operator[](name) = std::vector<std::string>();
        break;
      }
      case proto::AttrType::BLOCKS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to BLOCKS";
        attrs_ptr->operator[](name) = std::vector<BlockDesc *>();
        return;
      }
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported attribute type (code %d).", attr_type));
    }
    need_update_ = true;
    return;
  }

  // In order to set bool attr properly
  if (attr_type == proto::AttrType::INT) {
    if (HasProtoAttr(name) &&
        GetProtoAttr(name).type() == proto::AttrType::BOOLEAN) {
      attrs_ptr->operator[](name) = static_cast<bool>(PADDLE_GET_CONST(int, v));
      need_update_ = true;
      return;
    }
    if (extra_attr_iter != extra_attr_map.end() &&
        static_cast<proto::AttrType>(extra_attr_iter->second.index() - 1) ==
            proto::AttrType::BOOLEAN) {
      attrs_ptr->operator[](name) = static_cast<bool>(PADDLE_GET_CONST(int, v));
      need_update_ = true;
      return;
    }
  }

  attrs_ptr->operator[](name) = v;
  need_update_ = true;
}

void OpDesc::SetVarAttr(const std::string &name, VarDesc *var) {
  this->attrs_[name] = var;
  need_update_ = true;
}

void OpDesc::SetVarsAttr(const std::string &name, std::vector<VarDesc *> vars) {
  this->attrs_[name] = vars;
  need_update_ = true;
}

void OpDesc::SetBlockAttr(const std::string &name, BlockDesc *block) {
  this->attrs_[name] = block;
  need_update_ = true;
}

void OpDesc::SetBlocksAttr(const std::string &name,
                           std::vector<BlockDesc *> blocks) {
  this->attrs_[name] = blocks;
  need_update_ = true;
}

void OpDesc::SetAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  attrs_ = attr_map;
  need_update_ = true;
}

void OpDesc::SetRuntimeAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  runtime_attrs_ = attr_map;
  need_update_ = true;
}

Attribute OpDesc::GetAttr(const std::string &name, bool with_attr_var) const {
  auto it = attrs_.find(name);
  if (it == attrs_.end()) {
    it = runtime_attrs_.find(name);
  }
  PADDLE_ENFORCE_NE(
      it,
      attrs_.end(),
      platform::errors::NotFound("Attribute %s is not found.", name));
  if (!with_attr_var) {
    PADDLE_ENFORCE_EQ(
        HasAttrVar(it->second),
        false,
        platform::errors::NotFound(
            "Attribute %s with constant value is not found, but found it with "
            "Variable(s) type, which maybe not supported in some scenarios "
            "currently, such as TensorRT et.al",
            name));
  }
  return it->second;
}

const proto::OpProto::Attr &OpDesc::GetProtoAttr(
    const std::string &name) const {
  const proto::OpProto &proto = OpInfoMap::Instance().Get(Type()).Proto();
  for (int i = 0; i != proto.attrs_size(); ++i) {
    const proto::OpProto::Attr &attr = proto.attrs(i);
    if (attr.name() == name) {
      return attr;
    }
  }

  PADDLE_THROW(platform::errors::NotFound(
      "Attribute %s is not found in proto %s.", name, proto.type()));
}

Attribute OpDesc::GetNullableAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  if (it != attrs_.end()) {
    return it->second;
  } else {
    return Attribute();
  }
}

std::vector<int> OpDesc::GetBlocksAttrIds(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      attrs_.end(),
      platform::errors::NotFound(
          "Attribute `%s` is not found in operator `%s`.", name, desc_.type()));
  auto blocks = PADDLE_GET_CONST(std::vector<BlockDesc *>, it->second);

  std::vector<int> ids;
  for (auto n : blocks) {
    ids.push_back(n->ID());
  }

  return ids;
}

int OpDesc::GetBlockAttrId(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      attrs_.end(),
      platform::errors::NotFound(
          "Attribute `%s` is not found in operator `%s`.", name, desc_.type()));
  return PADDLE_GET_CONST(BlockDesc *, it->second)->ID();
}

const std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() const {
  return attrs_;
}

const AttributeMap &OpDesc::GetRuntimeAttrMap() const { return runtime_attrs_; }

void OpDesc::Rename(const std::string &old_name, const std::string &new_name) {
  RenameInput(old_name, new_name);
  RenameOutput(old_name, new_name);
  need_update_ = true;
}

void OpDesc::RenameOutput(const std::string &old_name,
                          const std::string &new_name) {
  for (auto &output : outputs_) {
    std::replace(
        output.second.begin(), output.second.end(), old_name, new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = PADDLE_GET(std::vector<std::string>, it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

void OpDesc::RenameInput(const std::string &old_name,
                         const std::string &new_name) {
  for (auto &input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = PADDLE_GET(std::vector<std::string>, it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

struct SetAttrDescVisitor {
  explicit SetAttrDescVisitor(proto::OpDesc::Attr *attr) : attr_(attr) {}
  mutable proto::OpDesc::Attr *attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(double v) const { attr_->set_float64(v); }
  void operator()(const std::string &v) const { attr_->set_s(v); }

  // Please refer to https://github.com/PaddlePaddle/Paddle/issues/7162
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type>
  void operator()(T b) const {
    attr_->set_b(b);
  }

  void operator()(const std::vector<int> &v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float> &v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string> &v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool> &v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }

  void operator()(const std::vector<VarDesc *> &v) const {
    std::vector<std::string> var_names;
    for (auto var : v) {
      var_names.emplace_back(var->Name());
    }
    VectorToRepeated(var_names, attr_->mutable_vars_name());
  }

  void operator()(const VarDesc *desc) const {
    attr_->set_var_name(desc->Name());
  }

  void operator()(const std::vector<BlockDesc *> &v) const {
    std::vector<int> blocks_idx;
    for (auto blk : v) {
      blocks_idx.push_back(blk->ID());
    }
    VectorToRepeated(blocks_idx, attr_->mutable_blocks_idx());
  }

  void operator()(BlockDesc *desc) const { attr_->set_block_idx(desc->ID()); }

  void operator()(int64_t v) const { attr_->set_l(v); }

  void operator()(const std::vector<int64_t> &v) const {
    VectorToRepeated(v, attr_->mutable_longs());
  }

  void operator()(const std::vector<double> &v) const {
    VectorToRepeated(v, attr_->mutable_float64s());
  }

  void operator()(paddle::blank) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method of SetAttrDescVisitor object for "
        "`boosst::blank` type."));
  }
};

void OpDesc::Flush() {
  VLOG(4) << "Flush "
          << " " << Type() << " " << need_update_;
  if (need_update_) {
    this->desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->desc_.mutable_attrs()->Clear();
    auto set_attr_desc = [this](const std::string &attr_name,
                                const Attribute &attr) -> void {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr_name);
      attr_desc->set_type(static_cast<proto::AttrType>(attr.index() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      paddle::visit(visitor, attr);
    };

    std::vector<std::pair<std::string, Attribute>> sorted_attrs{attrs_.begin(),
                                                                attrs_.end()};
    std::sort(
        sorted_attrs.begin(),
        sorted_attrs.end(),
        [](std::pair<std::string, Attribute> a,
           std::pair<std::string, Attribute> b) { return a.first < b.first; });

    for (auto &attr : sorted_attrs) {
      set_attr_desc(attr.first, attr.second);
    }
    for (auto &attr : runtime_attrs_) {
      set_attr_desc(attr.first, attr.second);
    }

    need_update_ = false;
  }
}

void OpDesc::CheckAttrs() {
  PADDLE_ENFORCE_EQ(Type().empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "CheckAttrs() can not be called before type is set."));
  auto *checker = OpInfoMap::Instance().Get(Type()).Checker();
  if (checker == nullptr) {
    // checker is not configured. That operator could be generated by Paddle,
    // not by users.
    return;
  }
  VLOG(10) << "begin to check attribute of " << Type();
  checker->Check(&attrs_);
  const auto &extra_attr_checkers =
      operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(Type());
  if (!extra_attr_checkers.empty()) {
    for (const auto &extra_checker : extra_attr_checkers) {
      extra_checker(&runtime_attrs_, false);
    }
  }
}

void OpDesc::InferShape(const BlockDesc &block) {
  try {
    VLOG(3) << "CompileTime infer shape on " << Type();
    auto &op_info = OpInfoMap::Instance().Get(this->Type());
    this->CheckAttrs();
    auto &infer_shape = op_info.infer_shape_;
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(infer_shape),
        true,
        platform::errors::NotFound(
            "Operator %s's infer_shape is not registered.", this->Type()));
    CompileTimeInferShapeContext ctx(*this, block);
    if (VLOG_IS_ON(10)) {
      std::ostringstream sout;
      auto inames = this->InputArgumentNames();
      sout << " From [";
      std::copy(inames.begin(),
                inames.end(),
                std::ostream_iterator<std::string>(sout, ", "));
      sout << "] to [";
      auto onames = this->OutputArgumentNames();
      std::copy(onames.begin(),
                onames.end(),
                std::ostream_iterator<std::string>(sout, ", "));
      sout << "]";
      VLOG(10) << sout.str();
    }
    infer_shape(&ctx);
  } catch (platform::EnforceNotMet &exception) {
    framework::AppendErrorOpHint(Type(), &exception);
    throw std::move(exception);
  } catch (...) {
    std::rethrow_exception(std::current_exception());
  }
}

void OpDesc::InferVarType(BlockDesc *block) const {
  // There are a few places that var type can be set.
  // When VarDesc is created, default set to LOD_TENSOR.
  // When output variable is created, default is default set to LOD_TENSOR.
  // We limit here to be the only place that operator defines its customized
  // var type inference. Hence, we don't do any "default" setting here.
  auto &info = OpInfoMap::Instance().Get(this->Type());
  if (info.infer_var_type_) {
    InferVarTypeContext context(this, block);
    info.infer_var_type_(&context);
  }
}

OperatorDistAttr *OpDesc::MutableDistAttr() {
  if (dist_attr_) {
    return dist_attr_.get();
  } else {
    dist_attr_.reset(new OperatorDistAttr(*this));
    return dist_attr_.get();
  }
}

void OpDesc::SetDistAttr(const OperatorDistAttr &dist_attr) {
  MutableDistAttr();
  *dist_attr_ = dist_attr;
}

void OpDesc::UpdateVarAttr(const std::string &name, const Attribute &attr) {
  auto attr_type = static_cast<proto::AttrType>(attr.index() - 1);
  auto type = GetAttrType(name, true);
  if (type == proto::AttrType::VAR) {
    PADDLE_ENFORCE_EQ(
        attr_type,
        type,
        platform::errors::InvalidArgument(
            "Required attr.type == proto::AttrType::VAR, but received %s",
            attr_type));
    auto *var_desc = PADDLE_GET_CONST(VarDesc *, attr);
    VLOG(3) << "Update AttrVar " << name << " with " << var_desc->Name();
    attrs_[name] = FindVarRecursive(var_desc->Name());
  } else if (type == proto::AttrType::VARS) {
    PADDLE_ENFORCE_EQ(
        attr_type,
        type,
        platform::errors::InvalidArgument(
            "Required attr.type == proto::AttrType::VARS, but received %s",
            attr_type));
    auto vars_desc = PADDLE_GET_CONST(std::vector<VarDesc *>, attr);
    std::vector<VarDesc *> new_val;
    for (auto &var_desc : vars_desc) {
      VLOG(3) << "Update AttrVars " << name << " with " << var_desc->Name();
      new_val.emplace_back(FindVarRecursive(var_desc->Name()));
    }
    attrs_[name] = std::move(new_val);
  }
}

VarDesc *OpDesc::FindVarRecursive(const std::string &name) {
  auto *cur_block = block_;
  while (cur_block != nullptr && cur_block->ID() >= 0) {
    auto *var = block_->FindVar(name);
    if (var != nullptr) {
      return var;
    }
    cur_block = cur_block->ParentBlock();
  }
  PADDLE_THROW(platform::errors::NotFound(
      "Not found Var(%s) from Block(%d) back into global Block.",
      name,
      block_->ID()));
}

CompileTimeInferShapeContext::CompileTimeInferShapeContext(
    const OpDesc &op, const BlockDesc &block)
    : op_(op), block_(block) {}

bool CompileTimeInferShapeContext::HasInput(const std::string &name) const {
  auto inputs = op_.Inputs(/*with_attr_var=*/true);
  if (inputs.find(name) == inputs.end()) {
    return false;
  }
  const std::vector<std::string> &input_names =
      op_.Input(name, /*with_attr_var=*/true);
  auto length = input_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(
      length,
      1UL,
      platform::errors::InvalidArgument("Input(%s) should have only one value, "
                                        "but it has %d values now.",
                                        name,
                                        length));
  return block_.HasVarRecursive(input_names[0]);
}

bool CompileTimeInferShapeContext::HasOutput(const std::string &name) const {
  if (op_.Outputs().find(name) == op_.Outputs().end()) {
    return false;
  }
  const std::vector<std::string> &output_names = op_.Output(name);
  auto length = output_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length,
                    1UL,
                    platform::errors::InvalidArgument(
                        "Output(%s) should have only one value, "
                        "but it has %d values now.",
                        name,
                        length));
  return block_.HasVarRecursive(output_names[0]);
}

bool CompileTimeInferShapeContext::HasAttr(const std::string &name) const {
  return op_.HasAttr(name, /*with_attr_var=*/false);
}

bool CompileTimeInferShapeContext::HasInputs(const std::string &name) const {
  auto inputs = op_.Inputs(/*with_attr_var=*/true);
  if (inputs.find(name) == inputs.end()) {
    return false;
  }
  const std::vector<std::string> &input_names =
      op_.Input(name, /*with_attr_var=*/true);
  if (input_names.empty()) {
    return false;
  }
  for (auto &input : input_names) {
    if (!block_.HasVarRecursive(input)) return false;
  }
  return true;
}

bool CompileTimeInferShapeContext::HasOutputs(const std::string &name,
                                              bool allow_null) const {
  if (op_.Outputs().find(name) == op_.Outputs().end()) {
    return false;
  }
  const std::vector<std::string> &output_names = op_.Output(name);
  if (output_names.empty()) {
    return false;
  }
  if (allow_null) {
    for (auto &output : output_names) {
      if (block_.HasVarRecursive(output)) return true;
    }
    return false;
  } else {
    for (auto &output : output_names) {
      if (!block_.HasVarRecursive(output)) return false;
    }
    return true;
  }
}

AttrReader CompileTimeInferShapeContext::Attrs() const {
  return AttrReader(op_.GetAttrMap(), op_.GetRuntimeAttrMap());
}

std::vector<std::string> CompileTimeInferShapeContext::Inputs(
    const std::string &name) const {
  return op_.Input(name, /*with_attr_var=*/true);
}

std::vector<std::string> CompileTimeInferShapeContext::Outputs(
    const std::string &name) const {
  return op_.Output(name);
}

std::vector<DDim> CompileTimeInferShapeContext::GetRepeatedDims(
    const std::string &name) const {
  auto var = block_.FindVarRecursive(name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound("Variable %s is not found.", name));
  std::vector<DDim> res;
  try {
    auto shapes = var->GetShapes();
    for (const auto &s : shapes) {
      res.push_back(s.empty() ? phi::make_ddim({0UL}) : phi::make_ddim(s));
    }
  } catch (...) {
    VLOG(5) << "GetRepeatedDim of variable " << name << " error.";
    std::rethrow_exception(std::current_exception());
  }
  return res;
}

void CompileTimeInferShapeContext::SetDim(const std::string &name,
                                          const DDim &dim) {
  block_.FindVarRecursive(name)->SetShape(vectorize(dim));
}

void CompileTimeInferShapeContext::SetRepeatedDims(
    const std::string &name, const std::vector<DDim> &dims) {
  auto var = block_.FindVarRecursive(name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound("Variable %s is not found.", name));
  std::vector<std::vector<int64_t>> dim_vec(dims.size());
  std::transform(dims.begin(), dims.end(), dim_vec.begin(), phi::vectorize<>);
  var->SetShapes(dim_vec);
}

bool CompileTimeInferShapeContext::IsRuntime() const { return false; }

bool CompileTimeInferShapeContext::IsRunMKLDNNKernel() const { return false; }

proto::VarType::Type CompileTimeInferShapeContext::GetVarType(
    const std::string &name) const {
  return block_.FindVarRecursive(name)->GetType();
}

std::vector<std::string> AttrVarNames(const Attribute &attr) {
  std::vector<std::string> vars_name;
  if (IsAttrVar(attr)) {
    vars_name.emplace_back(PADDLE_GET_CONST(VarDesc *, attr)->Name());
  } else if (IsAttrVars(attr)) {
    for (auto &iter : PADDLE_GET_CONST(std::vector<VarDesc *>, attr)) {
      vars_name.emplace_back(iter->Name());
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported Attribute value type `%s` for AttrVarNames",
        platform::demangle(attr.type().name())));
  }
  return vars_name;
}

}  // namespace framework
}  // namespace paddle
