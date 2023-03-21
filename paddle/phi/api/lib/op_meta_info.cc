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
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {

PADDLE_API void AssignTensorImpl(const Tensor& src, Tensor* dst) {
  PADDLE_ENFORCE_EQ(src.is_dense_tensor() && dst->is_dense_tensor(),
                    true,
                    phi::errors::Unavailable(
                        "Now only supported DenseTensor in Custom Operator."));
  PADDLE_ENFORCE_EQ(
      src.initialized(),
      true,
      phi::errors::Unavailable(
          "The Custom OpKernel calculate output is not initialized."));
  PADDLE_ENFORCE_EQ(dst->defined(),
                    true,
                    phi::errors::Unavailable(
                        "The Custom OpKernel origin output is not defined."));
  auto& dense_src = static_cast<const phi::DenseTensor&>(*src.impl());
  auto* dense_dst = static_cast<phi::DenseTensor*>(dst->impl().get());
  *dense_dst = dense_src;
}

////////////////////// Kernel Context //////////////////////

void CustomOpKernelContext::EmplaceBackInput(Tensor&& input) {
  size_t index = inputs_.size();
  inputs_.emplace_back(input);
  input_range_.emplace_back(std::make_pair(index, index + 1));
}

void CustomOpKernelContext::EmplaceBackInputs(
    const std::vector<Tensor>& inputs) {
  size_t index = inputs_.size();
  input_range_.emplace_back(std::make_pair(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
}

void CustomOpKernelContext::EmplaceBackOutput(Tensor&& output) {
  size_t index = outputs_.size();
  outputs_.emplace_back(output);
  output_range_.emplace_back(std::make_pair(index, index + 1));
}

void CustomOpKernelContext::EmplaceBackOutputs(
    const std::vector<Tensor>& outputs) {
  size_t index = outputs_.size();
  output_range_.emplace_back(std::make_pair(index, index + outputs.size()));
  outputs_.insert(outputs_.end(),
                  std::make_move_iterator(outputs.begin()),
                  std::make_move_iterator(outputs.end()));
}

void CustomOpKernelContext::EmplaceBackAttr(paddle::any attr) {
  attrs_.emplace_back(std::move(attr));
  VLOG(7) << "attrs_ No." << attrs_.size() - 1
          << " has value of type: " << attrs_[attrs_.size() - 1].type().name();
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

Tensor& CustomOpKernelContext::MutableInputAt(size_t idx) {
  return inputs_.at(idx);
}

Tensor* CustomOpKernelContext::MutableOutputAt(size_t idx) {
  return &(outputs_.at(idx));
}
std::vector<Tensor*> CustomOpKernelContext::MutableOutputBetweeen(size_t start,
                                                                  size_t end) {
  std::vector<Tensor*> rlt;
  for (size_t i = start; i < end; ++i) {
    rlt.emplace_back(&(outputs_.at(i)));
  }
  return rlt;
}

std::vector<Tensor> CustomOpKernelContext::OutputsBetweeen(size_t start,
                                                           size_t end) {
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

// handle inplace mechanism
// Find out non-inplace output tensors.
void CustomOpKernelContext::MapPlainOutputs(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    auto& input = inputs[in_idx];
    if (inplace_map.find(input) == inplace_map.end()) {
      continue;
    }
    auto out_iter = find(outputs.begin(), outputs.end(), inplace_map.at(input));
    PADDLE_ENFORCE(
        out_iter != outputs.end(),
        phi::errors::NotFound("Can't find the mapped value of %s, please check "
                              "the input of `Inplace` again and make "
                              "sure you registered your op accurately. ",
                              input));
    inplace_tensor_map_[in_idx] = distance(outputs.begin(), out_iter);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (std::any_of(
            inplace_tensor_map_.begin(),
            inplace_tensor_map_.end(),
            [i](std::unordered_map<size_t, size_t>::const_reference pair) {
              return pair.second == i;
            })) {
      continue;
    }
    size_t output_start_idx = output_range_[i].first;
    size_t output_end_idx = output_range_[i].second;
    for (size_t idx = output_start_idx; idx < output_end_idx; ++idx) {
      plain_outputs_.push_back(&outputs_[idx]);
    }
  }
  VLOG(4) << "Custom opertor update inplace input-output map successfully.";
}
// Assign input tensor to inplace output tensors.
void CustomOpKernelContext::AssignInplaceOutputs() {
  for (auto pair : inplace_tensor_map_) {
    size_t in_start_idx = input_range_[pair.first].first;
    size_t in_end_idx = input_range_[pair.first].second;
    size_t out_start_idx = output_range_[pair.second].first;
    size_t out_end_idx = output_range_[pair.second].second;
    size_t assign_tensor_size = in_end_idx - in_start_idx;
    PADDLE_ENFORCE(
        assign_tensor_size == out_end_idx - out_start_idx,
        phi::errors::OutOfRange("When assigning inplaced tensor, Input vector "
                                "size %d mismatch output vector size %d",
                                in_end_idx - in_start_idx,
                                out_end_idx - out_start_idx));
    for (size_t i = 0; i < assign_tensor_size; ++i) {
      AssignTensorImpl(inputs_[in_start_idx + i], &outputs_[out_start_idx + i]);
    }
    VLOG(4)
        << "Custom opertor update inplace input-output tensor successfully.";
  }
}
std::vector<Tensor*>* CustomOpKernelContext::AllMutablePlainOutput() {
  return &plain_outputs_;
}
std::unordered_map<size_t, size_t>
CustomOpKernelContext::GetInplaceTensorMap() {
  return inplace_tensor_map_;
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

//////////////// Op Meta Info Map /////////////////

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
      phi::errors::PreconditionNotMet(
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
      PADDLE_THROW(phi::errors::InvalidArgument(
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
  info_ptr_->Attrs(std::forward<std::vector<std::string>>(attrs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInplaceMap(
    std::unordered_map<std::string, std::string>&& inplace_map) {
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
  PADDLE_ENFORCE_EQ(
      index_,
      0UL,
      phi::errors::Unimplemented(
          "Currently, the InferDtypeFn setting of Grad Op is not supported, "
          "And backward Tensor `X@GRAD` will use the dtype of forward Tensor "
          "`X` by default."));
  info_ptr_->SetInferDtypeFn(std::forward<InferDtypeFunc>(func));
  return *this;
}
}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
// C-API to get global OpMetaInfoMap.
paddle::OpMetaInfoMap& PD_GetOpMetaInfoMap() {
  return paddle::OpMetaInfoMap::Instance();
}
#endif

#ifdef __cplusplus
}  // end extern "C"
#endif
