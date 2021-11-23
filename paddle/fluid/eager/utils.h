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

#pragma once

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"

#include "paddle/pten/api/all.h"

namespace egr {

/**
 * EagerUtils is utils used to do some static conversion or autograd
 * members access, this class is desinged to be a full static functional
 * utils class
 * **/

class EagerUtils {
 public:
  /**
   * We have to use autograd_meta and multi_autograd_meta to initialize
   * autograd_meta for tensor, since we can't init it in
   * egr::EagerTensor's
   * constructor (it's abstract class there)
   *
   * **/
  static AutogradMeta* autograd_meta(egr::EagerTensor* target);

  static std::vector<AutogradMeta*> multi_autograd_meta(
      std::vector<egr::EagerTensor>* targets);

  static std::pair<size_t, size_t> OutRankInfo(const egr::EagerTensor& target);

  static std::shared_ptr<GradNodeBase> grad_node(
      const egr::EagerTensor& target);

  // Set history is used to set backward info during forward process, it will
  // set forward var's autograd meta's grad node as current backward node.
  static void SetHistory(std::vector<AutogradMeta*>* autograd_metas,
                         const std::shared_ptr<GradNodeBase>& grad_node);
  static void SetHistory(AutogradMeta* autograd_meta,
                         const std::shared_ptr<GradNodeBase>& grad_node);

  // This is used for Set vector of tensors' rank
  static void SetOutRankWithSlot(std::vector<AutogradMeta*>* targets,
                                 size_t slot_id);
  static void SetOutRankWithSlot(AutogradMeta* target, size_t slot_id);

  // This method will return an AutogradMeta pointer unsafely.
  static AutogradMeta* unsafe_autograd_meta(const egr::EagerTensor& target);
  static std::vector<AutogradMeta*> unsafe_autograd_meta(
      std::vector<egr::EagerTensor>* targets);
};

}  // namespace egr
