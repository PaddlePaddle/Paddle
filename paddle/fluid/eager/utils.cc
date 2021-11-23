// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/api/all.h"

namespace egr {
/* ---- Tensor -> Var ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToVars(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToVar in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToVar(
      paddle::framework::proto::VarType_Type_LOD_TENSOR);
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToVars(
    const std::vector<egr::EagerTensor>& tensors) {
  // TODO(jiabin): No const cast here. We should call SyncToVar in Python_C
  // wrapper
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    const_cast<EagerTensor*>(&(tensors[i]))
        ->SyncToVar(paddle::framework::proto::VarType_Type_LOD_TENSOR);
    res.emplace_back(new EagerTensor(tensors[i]));
  }
  return res;
}

/* ---- VarBase -> Tensor ---- */
std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToTensors(
    const egr::EagerTensor& tensor) {
  // TODO(jiabin): No const cast here. We should call SyncToTensor in Python_C
  // wrapper
  const_cast<EagerTensor*>(&tensor)->SyncToTensor();
  return {std::make_shared<EagerTensor>(tensor)};
}

std::vector<std::shared_ptr<egr::EagerTensor>> EagerUtils::SyncToTensors(
    const std::vector<egr::EagerTensor>& tensors) {
  // TODO(jiabin): No const cast here. We should call SyncToTensor in Python_C
  // wrapper
  std::vector<std::shared_ptr<EagerTensor>> res;
  size_t num = tensors.size();
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    const_cast<EagerTensor*>(&(tensors[i]))->SyncToTensor();
    res.emplace_back(new EagerTensor(tensors[i]));
  }
  return res;
}

std::vector<std::shared_ptr<EagerTensor>> EagerUtils::ConstructDuplicableOutput(
    const size_t num) {
  std::vector<std::shared_ptr<EagerTensor>> res;
  res.reserve(num);
  for (size_t i = 0; i < num; i++) {
    res.emplace_back(
        new EagerTensor(egr::Controller::Instance().GenerateUniqueName()));
  }
  return res;
}

std::vector<egr::EagerTensor> EagerUtils::GetOutputs(
    const std::vector<std::shared_ptr<EagerTensor>>& outs) {
  std::vector<egr::EagerTensor> res;
  res.reserve(outs.size());
  for (const auto& out : outs) {
    PADDLE_ENFORCE_NOT_NULL(out.get(),
                            "Eager Tensor %s is null and cannot be copied. "
                            "We are tring to Get Output tensor from its "
                            "shared_ptr, this error may indicate some outputs "
                            "are nullptr",
                            out->name());
    res.emplace_back((*(out.get())));
  }
  return res;
}

egr::EagerTensor EagerUtils::GetOutput(
    const std::shared_ptr<EagerTensor>& out) {
  PADDLE_ENFORCE_NOT_NULL(out.get(),
                          "Eager Tensor %s is null and cannot be copied. We "
                          "are tring to Get Output tensor from its shared_ptr, "
                          "this error may indicate output is nullptr",
                          out->name());
  return EagerTensor((*(out.get())));
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(const egr::EagerTensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta(), "
                     "if you are using unsafe_autograd_meta, please make sure "
                     "your tensor's autograd_meta is set"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(
    const egr::EagerTensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}
}  // namespace egr
