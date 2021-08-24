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

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/function_api.h"

#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/dtype.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

/**
 * Implementation of GradNodeBase, Edge and InputBuffer.
**/
namespace egr {

void GradNodeBase::AddEdges(const std::vector<AutogradMeta*>& metas) {
  VLOG(0) << "Add Edge for tensors";
  for (const auto& meta : metas) {
    adj_edges_.emplace_back(meta->GetMutableGradNode(), meta->OutRank());
  }
}

const std::vector<Edge>& GradNodeBase::GetEdges() const { return adj_edges_; }

void GradNodeBase::RecordStopGradient(
    const std::vector<AutogradMeta*>& ins_autograds) {
  for (size_t i = 0; i < ins_autograds.size(); ++i) {
    bwd_stop_gradients_.emplace_back(std::move(ins_autograds[i]->NumericStopGradient()));
  }
}
  
void GradNodeBase::RegisterGradientHook(size_t output_rank, const std::function<pt::Tensor(const pt::Tensor&)>& hook) {
    gradient_hooks_.push_back(std::make_pair(output_rank, hook));
}
  
void GradNodeBase::RegisterReduceHook(const std::function<void(void)>& hook) {
    reduce_hooks_.push_back(hook);
}
  
std::vector<pt::Tensor> GradNodeBase::ApplyGradientHooks(const std::vector<pt::Tensor>& tensors) {
    std::vector<pt::Tensor> outs(tensors.size());
    for(auto& pair : gradient_hooks_) {
        size_t output_rank = pair.first;
        std::function<pt::Tensor(const pt::Tensor&)>& hook = pair.second;
    
        PADDLE_ENFORCE(output_rank < tensors.size(), 
            paddle::platform::errors::Fatal("OutputRank from registered hook should be smaller than size of grad_tensors"));

        pt::Tensor& out = outs[output_rank];
        if(!out.defined() || !out.initialized()) {
            out = hook(tensors[output_rank]);
        } else {
            out = hook(out);
        }
    }

    for(size_t i = 0; i < outs.size(); i++) {
        if(!outs[i].defined() || !outs[i].initialized()) {
            outs[i] = tensors[i];
        }
    }

    return outs;
}
  
void GradNodeBase::ApplyReduceHooks() {
    for(auto& hook : reduce_hooks_) {
        hook();
    }
}

void InputBuffer::add(size_t pos, const pt::Tensor& t, bool fill_one) {
    PADDLE_ENFORCE(pos < buffer.size(),
        paddle::platform::errors::Fatal("Invalid pos for InputBuffer::add() which exceeds size of buffer"));
    pt::Tensor& buffer_tensor = buffer[pos];
    if(!fill_one) {
        if(!buffer_tensor.defined() || !buffer_tensor.initialized()) {
            // Simply copy tensor->impl
            buffer_tensor = t;

        } else {
            // Accumulation
            AccumulateTensorsAPI(buffer_tensor, t);
        }

    } else {
        // Create new tensor->impl and fill it with 1.0
        auto t_impl = t.impl();

        // Fill 1.0
        FillConstAPI(1.0, t_impl->dims(), t_impl->backend(), t_impl->type(), t_impl->layout(), buffer_tensor);
    }
}

}  // namespace egr
