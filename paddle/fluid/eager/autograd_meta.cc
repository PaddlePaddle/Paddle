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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/eager/nodes/accumulation_node.h"

/**
 * Implementation of AutogradMeta and AbstractAutogradMeta.
**/

namespace egr {
  
AutogradMeta* EagerUtils::autograd_meta(pt::Tensor& target) {
  auto* p_autograd_meta = target.get_autograd_meta();
  if (!p_autograd_meta) {
    auto p_autograd_meta_ptr = std::make_shared<AutogradMeta>();
    p_autograd_meta = p_autograd_meta_ptr.get();
    target.set_autograd_meta(p_autograd_meta_ptr);
  }
  return static_cast<AutogradMeta*>(p_autograd_meta);
}

std::vector<AutogradMeta*> EagerUtils::multi_autograd_meta(
    const std::vector<pt::Tensor>& targets) {
  std::vector<AutogradMeta*> ret;

  // for multi_autograd_meta we can tolerent it has nullptr.
  for (const auto& t : targets) {
    auto* p_autograd_meta = t.get_autograd_meta();
    ret.push_back(static_cast<AutogradMeta*>(p_autograd_meta));
  }
  return ret;
}

size_t EagerUtils::output_rank(pt::Tensor& target) {
  return autograd_meta(target)->OutRank();
}

std::shared_ptr<GradNodeBase> EagerUtils::grad_node(pt::Tensor& target) {
  return autograd_meta(target)->GetMutableGradNode();
}

bool EagerUtils::IsLeafTensor(pt::Tensor& target) {
    std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(target);
    if(std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node)) {
        return true;
    }

    return false;
}

void EagerUtils::PassStopGradient(AutogradMeta** outs, size_t outs_num,
                      bool generate_grad) {
  for (size_t i = 0; i < outs_num; ++i) {
    if (!outs[i]) {
      // TODO(jiabin): Add Tensor name here when we supported.
      VLOG(0) << "Tensor is NULL";
      continue;
    }
    outs[i]->SetNumericStopGradient(generate_grad);
  }
}

bool EagerUtils::ComputeRequireGrad(AutogradMeta** ins, size_t ins_num, AutogradMeta** outs,
                        size_t outs_num, bool trace_backward) {
  if (!trace_backward) return false;

  for (size_t i = 0; i < ins_num; ++i) {
    auto ins_stop_gradient = ins[i]->StopGradient();
    if (!ins_stop_gradient) {
      VLOG(0) << "Find out input Stop Gradient is False";
      PassStopGradient(outs, outs_num, ins_stop_gradient);
      return true;
    }
  }
  return false;
}

void EagerUtils::SetHistoryForTensor(pt::Tensor& target, const std::shared_ptr<GradNodeBase>& grad_node) {
  AutogradMeta* autograd_meta = EagerUtils::autograd_meta(target);
  autograd_meta->SetGradNode(grad_node);
}
  
pt::Tensor EagerUtils::CreateTensorWithValue(const pt::DDim& ddim, const pt::Backend& backend,
                                             const pt::DataType& dtype, const pt::DataLayout& layout,
                                             double value, bool is_leaf) {
    pt::Tensor out = pt::Tensor();
    FillConstAPI(value, ddim, backend, dtype, layout, out);
    
    if(is_leaf) {
        AutogradMeta* meta = autograd_meta(out);
        auto accumulation_node = std::make_shared<GradNodeAccumulation>();
        meta->SetGradNode(accumulation_node);
    }

    return out;
}

}  // namespace egr
