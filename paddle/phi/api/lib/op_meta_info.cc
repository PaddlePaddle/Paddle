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

#include "paddle/phi/api/ext/op_meta_info.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {

// remove leading and tailing spaces
std::string trim_spaces(const std::string& str) {
  const char* p = str.c_str();
  while (*p != 0 && isspace(*p)) {
    p++;
  }
  size_t len = strlen(p);
  while (len > 0 && isspace(p[len - 1])) {
    len--;
  }
  return std::string(p, len);
}

std::vector<std::string> ParseAttrStr(const std::string& attr) {
  auto split_pos = attr.find_first_of(':');
  PADDLE_ENFORCE_NE(split_pos,
                    std::string::npos,
                    common::errors::InvalidArgument(
                        "Invalid attribute string format. Attribute string "
                        "format is `<name>:<type>`."));

  std::vector<std::string> rlt;
  // 1. name
  rlt.emplace_back(trim_spaces(attr.substr(0, split_pos)));
  // 2. type
  rlt.emplace_back(trim_spaces(attr.substr(split_pos + 1)));

  VLOG(3) << "attr name: " << rlt[0] << ", attr type str: " << rlt[1];

  return rlt;
}

PADDLE_API void AssignTensorImpl(const Tensor& src, Tensor* dst) {
  if (!src.has_allocation() || !dst->defined()) {
    VLOG(3) << "Custom operator assigns non-initialized tensor, this only "
               "happens when handling inplace optional inputs & outputs.";
    return;
  }
  PADDLE_ENFORCE_EQ(
      ((src.is_dense_tensor() && dst->is_dense_tensor()) ||
       (src.is_dist_tensor() && dst->is_dist_tensor())),
      true,
      common::errors::Unavailable(
          "Now only supported DenseTensor and DistTensor in Custom Operator."));
  PADDLE_ENFORCE_EQ(
      src.has_allocation(),
      true,
      common::errors::Unavailable(
          "The Custom OpKernel calculate output is not initialized."));
  PADDLE_ENFORCE_EQ(dst->defined(),
                    true,
                    common::errors::Unavailable(
                        "The Custom OpKernel origin output is not defined."));
  if (src.is_dense_tensor()) {
    auto& dense_src = static_cast<const phi::DenseTensor&>(*src.impl());
    auto* dense_dst = static_cast<phi::DenseTensor*>(dst->impl().get());
    *dense_dst = dense_src;
  } else {
    auto* dense_src =
        static_cast<phi::distributed::DistTensor*>(src.impl().get())
            ->unsafe_mutable_value();
    auto* dense_dst =
        static_cast<phi::distributed::DistTensor*>(dst->impl().get())
            ->unsafe_mutable_value();
    *dense_dst = *dense_src;
  }
}

////////////////////// Kernel Context //////////////////////

void CustomOpKernelContext::EmplaceBackInput(Tensor&& input) {
  size_t index = inputs_.size();
  inputs_.emplace_back(input);
  input_range_.emplace_back(index, index + 1);
}

void CustomOpKernelContext::EmplaceBackInput(const Tensor& input) {
  size_t index = inputs_.size();
  inputs_.emplace_back(input);
  input_range_.emplace_back(index, index + 1);
}

void CustomOpKernelContext::EmplaceBackInputs(
    const std::vector<Tensor>& inputs) {
  size_t index = inputs_.size();
  input_range_.emplace_back(index, index + inputs.size());
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}

void CustomOpKernelContext::EmplaceBackOutput(Tensor&& output) {
  size_t index = outputs_.size();
  outputs_.emplace_back(output);
  output_range_.emplace_back(index, index + 1);
}

void CustomOpKernelContext::EmplaceBackOutputs(
    const std::vector<Tensor>& outputs) {
  size_t index = outputs_.size();
  output_range_.emplace_back(index, index + outputs.size());
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

void CustomOpKernelContext::EmplaceBackAttr(paddle::any attr) {
  attrs_.emplace_back(std::move(attr));
  VLOG(7) << "attrs_ No." << attrs_.size() - 1
          << " has value of type: " << attrs_[attrs_.size() - 1].type().name();
}

void CustomOpKernelContext::EmplaceBackAttrs(
    const std::vector<paddle::any>& attrs) {
  attrs_ = attrs;
}

const Tensor& CustomOpKernelContext::InputAt(size_t idx) const {
  return inputs_.at(idx);
}

std::vector<Tensor> CustomOpKernelContext::InputsBetween(size_t start,
                                                         size_t end) const {
  std::vector<Tensor> rlt;
  for (size_t i = start; i < end; ++i) {
    rlt.emplace_back(inputs_.at(i));
  }
  return rlt;
}

const std::vector<paddle::any>& CustomOpKernelContext::Attrs() const {
  return attrs_;
}

Tensor& CustomOpKernelContext::MutableInputAt(size_t idx) {
  return inputs_.at(idx);
}

std::vector<Tensor>* CustomOpKernelContext::AllMutableInput() {
  return &inputs_;
}

paddle::optional<Tensor> CustomOpKernelContext::OptionalInputAt(
    size_t idx) const {
  if (!inputs_.at(idx).initialized()) {
    return paddle::none;
  }
  return paddle::make_optional<paddle::Tensor>(inputs_.at(idx));
}

paddle::optional<std::vector<Tensor>>
CustomOpKernelContext::OptionalInputsBetween(size_t start, size_t end) const {
  std::vector<Tensor> rlt;
  for (size_t i = start; i < end; ++i) {
    if (!inputs_.at(i).initialized()) {
      return paddle::none;
    }
    rlt.emplace_back(inputs_.at(i));
  }
  return paddle::make_optional<std::vector<Tensor>>(rlt);
}

Tensor* CustomOpKernelContext::MutableOutputAt(size_t idx) {
  return &(outputs_.at(idx));
}
std::vector<Tensor*> CustomOpKernelContext::MutableOutputBetween(size_t start,
                                                                 size_t end) {
  std::vector<Tensor*> rlt;
  for (size_t i = start; i < end; ++i) {
    rlt.emplace_back(&(outputs_.at(i)));
  }
  return rlt;
}

std::vector<Tensor> CustomOpKernelContext::OutputsBetween(size_t start,
                                                          size_t end) const {
  std::vector<Tensor> rlt;
  for (size_t i = start; i < end; ++i) {
    rlt.emplace_back(outputs_.at(i));
  }
  return rlt;
}

std::vector<Tensor>* CustomOpKernelContext::AllMutableOutput() {
  return &outputs_;
}

const std::pair<size_t, size_t>& CustomOpKernelContext::InputRangeAt(
    size_t idx) const {
  return input_range_.at(idx);
}
const std::pair<size_t, size_t>& CustomOpKernelContext::OutputRangeAt(
    size_t idx) const {
  return output_range_.at(idx);
}

const std::vector<std::pair<size_t, size_t>>&
CustomOpKernelContext::InputRange() const {
  return input_range_;
}

const std::vector<std::pair<size_t, size_t>>&
CustomOpKernelContext::OutputRange() const {
  return output_range_;
}

void CustomOpKernelContext::ConstructInplaceIndex(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  // Cache inplace indices.
  if (inplace_map.empty() || !inplace_idx_map_.empty()) {
    VLOG(4) << "Custom operator ConstructInplaceIndex no need to recompute.";
    return;
  }
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    auto& input = inputs[in_idx];
    if (inplace_map.find(input) == inplace_map.end()) {
      continue;
    }
    auto out_iter = find(outputs.begin(), outputs.end(), inplace_map.at(input));
    PADDLE_ENFORCE_NE(out_iter,
                      outputs.end(),
                      common::errors::NotFound(
                          "Can't find the mapped value of %s, please check "
                          "the input of `Inplace` again and make "
                          "sure you registered your op accurately. ",
                          input));
    size_t out_idx = distance(outputs.begin(), out_iter);
    inplace_idx_map_[in_idx] = out_idx;
    inplace_reverse_idx_map_[out_idx] = in_idx;
  }
  VLOG(4) << "Custom operator update inplace input-output map successfully.";
}

// Find out non-inplace output tensors.
void CustomOpKernelContext::UpdatePlainOutputs(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  // Cache plain outputs vector.
  if (!plain_outputs_.empty()) {
    VLOG(4) << "Custom operator UpdatePlainOutputs no need to recompute.";
    return;
  }
  ConstructInplaceIndex(inputs, outputs, inplace_map);
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (inplace_reverse_idx_map_.find(i) != inplace_reverse_idx_map_.end()) {
      continue;
    }
    size_t output_start_idx = output_range_[i].first;
    size_t output_end_idx = output_range_[i].second;
    for (size_t idx = output_start_idx; idx < output_end_idx; ++idx) {
      plain_outputs_.push_back(&outputs_[idx]);
    }
  }
  VLOG(4) << "Custom operator update plain outputs map successfully.";
}

// Assign input tensor to inplace output tensors.
void CustomOpKernelContext::AssignInplaceOutputs() {
  for (auto pair : inplace_idx_map_) {
    size_t in_start_idx = input_range_[pair.first].first;
    size_t in_end_idx = input_range_[pair.first].second;
    size_t out_start_idx = output_range_[pair.second].first;
    size_t out_end_idx = output_range_[pair.second].second;
    size_t assign_tensor_size = in_end_idx - in_start_idx;
    PADDLE_ENFORCE_EQ(assign_tensor_size,
                      out_end_idx - out_start_idx,
                      common::errors::OutOfRange(
                          "When assigning inplaced tensor, Input vector "
                          "size %d mismatch output vector size %d",
                          in_end_idx - in_start_idx,
                          out_end_idx - out_start_idx));
    for (size_t i = 0; i < assign_tensor_size; ++i) {
      AssignTensorImpl(inputs_[in_start_idx + i], &outputs_[out_start_idx + i]);
    }
    VLOG(4) << "Custom operator update inplace input-output tensor "
               "successfully. Update map size = "
            << inplace_idx_map_.size();
  }
}

std::vector<Tensor*>* CustomOpKernelContext::AllMutablePlainOutput() {
  return &plain_outputs_;
}

std::unordered_map<size_t, size_t> CustomOpKernelContext::GetInplaceIndexMap()
    const {
  return inplace_idx_map_;
}

std::unordered_map<size_t, size_t>
CustomOpKernelContext::GetInplaceReverseIndexMap() const {
  return inplace_reverse_idx_map_;
}
////////////////////// Op Meta Info //////////////////////

OpMetaInfo& OpMetaInfo::Inputs(std::vector<std::string>&& inputs) {
  inputs_ = std::forward<std::vector<std::string>>(inputs);
  return *this;
}
OpMetaInfo& OpMetaInfo::Outputs(std::vector<std::string>&& outputs) {
  outputs_ = std::forward<std::vector<std::string>>(outputs);
  return *this;
}
OpMetaInfo& OpMetaInfo::Attrs(std::vector<std::string>&& attrs) {
  attrs_ = std::forward<std::vector<std::string>>(attrs);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInplaceMap(
    std::unordered_map<std::string, std::string>&& inplace_map) {
  inplace_map_ =
      std::forward<std::unordered_map<std::string, std::string>>(inplace_map);
  for (const auto& pair : inplace_map_) {
    inplace_reverse_map_[pair.second] = pair.first;
  }
  return *this;
}
OpMetaInfo& OpMetaInfo::SetKernelFn(KernelFunc&& func) {
  kernel_fn_ = std::forward<KernelFunc>(func);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferShapeFn(InferShapeFunc&& func) {
  infer_shape_fn_ = std::forward<InferShapeFunc>(func);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferDtypeFn(InferDtypeFunc&& func) {
  infer_dtype_fn_ = std::forward<InferDtypeFunc>(func);
  return *this;
}

OpMetaInfo& OpMetaInfo::SetInferSpmdFn(InferSpmdFunc&& func) {
  infer_spmd_fn_ = std::forward<InferSpmdFunc>(func);
  return *this;
}

bool OpMetaInfo::IsDoubleGradOp() const {
  if (name_.find("_grad_grad") != name_.npos) {
    return true;
  }
  return false;
}

bool OpMetaInfo::IsGradOp() const {
  if (!IsDoubleGradOp() && name_.find("_grad") != name_.npos) {
    return true;
  }
  return false;
}

#ifdef PADDLE_WITH_TENSORRT
OpMetaInfo& OpMetaInfo::SetTrtInferShapeFn(TrtGetOutputDimsFunc&& func) {
  trt_infer_shape_fn_ = std::forward<TrtGetOutputDimsFunc>(func);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetTrtSupportsFormatConfig(
    std::vector<std::string>&& config) {
  trt_supports_format_config_ = std::forward<std::vector<std::string>>(config);
  return *this;
}
#endif

//////////////// Op Meta Info Helper /////////////////
const std::string& OpMetaInfoHelper::GetOpName(const paddle::OpMetaInfo& info) {
  return info.name_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetInputs(
    const paddle::OpMetaInfo& info) {
  return info.inputs_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetOutputs(
    const paddle::OpMetaInfo& info) {
  return info.outputs_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetAttrs(
    const paddle::OpMetaInfo& info) {
  return info.attrs_;
}
const std::unordered_map<std::string, std::string>&
OpMetaInfoHelper::GetInplaceMap(const paddle::OpMetaInfo& info) {
  return info.inplace_map_;
}
const std::unordered_map<std::string, std::string>&
OpMetaInfoHelper::GetInplaceReverseMap(const paddle::OpMetaInfo& info) {
  return info.inplace_reverse_map_;
}
const KernelFunc& OpMetaInfoHelper::GetKernelFn(
    const paddle::OpMetaInfo& info) {
  return info.kernel_fn_;
}
const InferShapeFunc& OpMetaInfoHelper::GetInferShapeFn(
    const paddle::OpMetaInfo& info) {
  return info.infer_shape_fn_;
}
const InferDtypeFunc& OpMetaInfoHelper::GetInferDtypeFn(
    const paddle::OpMetaInfo& info) {
  return info.infer_dtype_fn_;
}

const InferSpmdFunc& OpMetaInfoHelper::GetInferSpmdFn(
    const paddle::OpMetaInfo& info) {
  return info.infer_spmd_fn_;
}

#ifdef PADDLE_WITH_TENSORRT
const TrtGetOutputDimsFunc& OpMetaInfoHelper::GetTrtInferShapeFn(
    const paddle::OpMetaInfo& info) {
  return info.trt_infer_shape_fn_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetTrtSupportsFormatConfig(
    const paddle::OpMetaInfo& info) {
  return info.trt_supports_format_config_;
}
#endif

//////////////// Op Meta Info Map /////////////////

OpMetaInfoMap& OpMetaInfoMap::Instance() {
  static OpMetaInfoMap g_custom_op_meta_info_map;
  return g_custom_op_meta_info_map;
}

std::vector<OpMetaInfo>& OpMetaInfoMap::operator[](const std::string& name) {
  return map_[name];
}

const std::unordered_map<std::string, std::vector<OpMetaInfo>>&
OpMetaInfoMap::GetMap() const {
  return map_;
}

//////////////// Op Meta Info Builder /////////////////

OpMetaInfoBuilder::OpMetaInfoBuilder(std::string&& name, size_t index) {
  // 1. member assign
  name_ = std::forward<std::string>(name);
  index_ = index;

  // 2. check and meta info build
  auto& info_vector = OpMetaInfoMap::Instance()[name_];
  // index check
  PADDLE_ENFORCE_EQ(
      info_vector.size(),
      index_,
      common::errors::PreconditionNotMet(
          "The operator %s's meta info register failed. "
          "Please make sure you call marcos as order `PD_BUILD_OP`, "
          "`PD_BUILD_GRAD_OP`, `PD_BUILD_DOUBLE_GRAD_OP`.",
          name_));
  switch (index_) {
    case 0:
      break;
    case 1:
      name_ = name_ + "_grad";
      break;
    case 2:
      name_ = name_ + "_grad_grad";
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Not support index `%d` when construct OpMetaInfoBuilder, "
          "now only support `0, 1, 2`.",
          index_));
  }
  auto op_meta = OpMetaInfo(name_);
  info_vector.emplace_back(std::move(op_meta));
  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

OpMetaInfoBuilder& OpMetaInfoBuilder::Inputs(
    std::vector<std::string>&& inputs) {
  info_ptr_->Inputs(std::forward<std::vector<std::string>>(inputs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::Outputs(
    std::vector<std::string>&& outputs) {
  info_ptr_->Outputs(std::forward<std::vector<std::string>>(outputs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::Attrs(std::vector<std::string>&& attrs) {
  const std::unordered_set<std::string> custom_attrs_type(
      {"bool",
       "int",
       "float",
       "double",
       "int64_t",
       "std::string",
       "std::vector<int>",
       "std::vector<float>",
       "std::vector<int64_t>",
       "std::vector<std::string>"});
  for (const auto& attr : attrs) {
    auto attr_type_str = ParseAttrStr(attr)[1];
    if (custom_attrs_type.find(attr_type_str) == custom_attrs_type.end()) {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, `double`,  "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, "
          "Please check whether the attribute data type and "
          "data type string are matched.",
          attr_type_str));
    }
  }
  info_ptr_->Attrs(std::forward<std::vector<std::string>>(attrs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInplaceMap(
    std::unordered_map<std::string, std::string>&& inplace_map) {
  const std::vector<std::string>& inputs =
      OpMetaInfoHelper::GetInputs(*info_ptr_);
  const std::vector<std::string>& outputs =
      OpMetaInfoHelper::GetOutputs(*info_ptr_);
  for (const auto& pair : inplace_map) {
    PADDLE_ENFORCE_NE(
        std::find(inputs.begin(), inputs.end(), pair.first),
        inputs.cend(),
        common::errors::PreconditionNotMet(
            "The register of operator %s's `SetInplaceMap` failed. "
            "Please make sure: 1. Call `Inputs` and `Outputs` before "
            "`SetInplaceMap`; 2. The keys of inplace_map are inside `Inputs`",
            name_));
    PADDLE_ENFORCE_NE(
        std::find(outputs.begin(), outputs.end(), pair.second),
        outputs.cend(),
        common::errors::PreconditionNotMet(
            "The register of operator %s's `SetInplaceMap` failed. "
            "Please make sure: 1. Call `Inputs` and `Outputs` "
            "before `SetInplaceMap`; 2. The values of inplace_map "
            "are inside `Outputs`",
            name_));
  }
  info_ptr_->SetInplaceMap(
      std::forward<std::unordered_map<std::string, std::string>>(inplace_map));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetKernelFn(KernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferShapeFn(InferShapeFunc func) {
  info_ptr_->SetInferShapeFn(std::forward<InferShapeFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferDtypeFn(InferDtypeFunc func) {
  info_ptr_->SetInferDtypeFn(std::forward<InferDtypeFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferSpmdFn(InferSpmdFunc func) {
  info_ptr_->SetInferSpmdFn(std::forward<InferSpmdFunc>(func));
  return *this;
}

#ifdef PADDLE_WITH_TENSORRT
OpMetaInfoBuilder& OpMetaInfoBuilder::SetTrtInferShapeFn(
    TrtGetOutputDimsFunc func) {
  info_ptr_->SetTrtInferShapeFn(std::forward<TrtGetOutputDimsFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetTrtSupportsFormatConfig(
    std::vector<std::string>&& config) {
  info_ptr_->SetTrtSupportsFormatConfig(
      std::forward<std::vector<std::string>>(config));
  return *this;
}
#endif
}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
// C-API to get global OpMetaInfoMap.
paddle::OpMetaInfoMap* PD_GetOpMetaInfoMap() {
  return &paddle::OpMetaInfoMap::Instance();
}
#endif

#ifdef __cplusplus
}  // end extern "C"
#endif
