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
#include "paddle/fluid/eager/api/utils/global_utils.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/variable.h"

namespace egr {
/**
 * Implementation of Eager Utils.
**/

AutogradMeta* EagerUtils::autograd_meta(egr::EagerTensor* target) {
  auto* p_autograd_meta = target->get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target->set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

AutogradMeta* EagerUtils::unsafe_autograd_meta(const egr::EagerTensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  PADDLE_ENFORCE(p_autograd_meta,
                 paddle::platform::errors::Fatal(
                     "Null autograd_meta gotten from unsafe_autograd_meta()"));
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::unsafe_autograd_meta(
    std::vector<egr::EagerTensor>* targets) {
  std::vector<AutogradMeta*> metas;
  for (const egr::EagerTensor& t : *targets) {
    metas.push_back(unsafe_autograd_meta(t));
  }
  return metas;
}

std::vector<AutogradMeta*> EagerUtils::multi_autograd_meta(
    std::vector<egr::EagerTensor>* targets) {
  std::vector<AutogradMeta*> ret;
  ret.reserve(targets->size());

  // for multi_autograd_meta we can tolerent it has nullptr.
  for (auto& t : (*targets)) {
    auto* p_autograd_meta = autograd_meta(&t);
    ret.push_back(static_cast<AutogradMeta*>(p_autograd_meta));
  }
  return ret;
}

std::pair<size_t, size_t> EagerUtils::OutRankInfo(
    const egr::EagerTensor& target) {
  return unsafe_autograd_meta(target)->OutRankInfo();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(
    const egr::EagerTensor& target) {
  return unsafe_autograd_meta(target)->GetMutableGradNode();
}

void EagerUtils::SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
  for (const auto& autograd_meta : *autograd_metas) {
    autograd_meta->SetGradNode(grad_node);
  }
}

void EagerUtils::SetHistory(AutogradMeta* autograd_meta,
                            const std::shared_ptr<GradNodeBase>& grad_node) {
  autograd_meta->SetGradNode(grad_node);
}

void EagerUtils::SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                    size_t slot_id) {
  // Set OutRankInfo from 0 to size of targets
  for (size_t i = 0; i < targets->size(); i++) {
    (*targets)[i]->SetSingleOutRankWithSlot(slot_id, i);
  }
}
void EagerUtils::SetOutRankWithSlot(AutogradMeta* target, size_t slot_id) {
  target->SetSingleOutRankWithSlot(slot_id, 0);
}

}  // namespace egr
