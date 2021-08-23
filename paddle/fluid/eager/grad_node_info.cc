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

        const pt::Tensor& tensor = tensors[output_rank];
        outs[output_rank] = hook(tensor);
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

template<typename T>
static void add_kernel(const pt::DenseTensor& t0, const pt::DenseTensor& t1, pt::DenseTensor& out) {
    const T* t0_ptr = t0.data<T>();
    const T* t1_ptr = t1.data<T>();
    T* out_ptr = out.mutable_data<T>();
    for(int i = 0; i < t0.numel(); i++) {
        out_ptr[i] = t0_ptr[i] + t1_ptr[i];
    }
}

void InputBuffer::add(size_t pos, const pt::Tensor& t, bool fill_one) {
    // TODO: Add support for other tensor types
    std::shared_ptr<pt::DenseTensor> tensor_instance = std::dynamic_pointer_cast<pt::DenseTensor>(t.impl());
    
    PADDLE_ENFORCE(tensor_instance != nullptr, 
        paddle::platform::errors::Fatal("InputBuffer::add() Only supports DenseTensor for now."));
    
    PADDLE_ENFORCE(tensor_instance->backend() == pt::Backend::kCPU, 
        paddle::platform::errors::Fatal("InputBuffer::add() Only supports tensors with CPU backend for now."));
    
    PADDLE_ENFORCE(pos < buffer.size(), 
        paddle::platform::errors::Fatal("Invalid pos for InputBuffer::add() which exceeds size of buffer"));
    
    pt::Tensor& buffer_tensor = buffer[pos];
    if(!fill_one) {
        if(!buffer_tensor.defined() || !buffer_tensor.initialized()) {
            // Simply copy tensor->impl
            buffer_tensor = t;

        } else {
            // Accumulation
            std::shared_ptr<pt::DenseTensor> buffer_tensor_instance = std::dynamic_pointer_cast<pt::DenseTensor>(buffer_tensor.impl());
            
            PADDLE_ENFORCE(buffer_tensor_instance != nullptr, 
                paddle::platform::errors::Fatal("InputBuffer::add() Only supports InputBuffer with DenseTensor for now."));
            PADDLE_ENFORCE(t.type() == buffer_tensor.type(), 
                paddle::platform::errors::Fatal("Unable to accumulate tensors with different dtype"));
            PADDLE_ENFORCE(t.numel() == buffer_tensor.numel(), 
                paddle::platform::errors::Fatal("Unable to accumulate tensors with different sizes"));
            
            PADDLE_ENFORCE(t.initialized(), 
                paddle::platform::errors::Fatal("Tensors to accumulate has not been initialized"));
    
            
            // TODO: Replace this with call to add_kernel_api
            switch(t.type()) {
                case pt::DataType::kINT64: {
                    add_kernel<int64_t>(*buffer_tensor_instance.get(), *tensor_instance.get(), *buffer_tensor_instance.get());
                    break;
                }
                case pt::DataType::kINT32: {
                    add_kernel<int32_t>(*buffer_tensor_instance.get(), *tensor_instance.get(), *buffer_tensor_instance.get());
                    break;
                }
                case pt::DataType::kFLOAT64: {
                    add_kernel<double>(*buffer_tensor_instance.get(), *tensor_instance.get(), *buffer_tensor_instance.get());
                    break;
                }
                case pt::DataType::kFLOAT32: {
                    add_kernel<float>(*buffer_tensor_instance.get(), *tensor_instance.get(), *buffer_tensor_instance.get());
                    break;
                }
                default: {
                    PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
                    break;
                }
            }
        }

    } else {
        // Create new tensor->impl and fill it with 1.0
        std::unique_ptr<pt::TensorMeta> tensor_meta = std::make_unique<pt::TensorMeta>(tensor_instance->dims(), tensor_instance->backend(), 
                                                                                       tensor_instance->type(), tensor_instance->layout());
        
        // Fill 1.0
        std::shared_ptr<pt::DenseTensor> tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
        switch(tensor_dense->type()) {
            case pt::DataType::kINT64: {
                int64_t* data_ptr = tensor_dense->mutable_data<int64_t>();
                for(int i = 0; i < tensor_dense->numel(); i++)
                    data_ptr[i] = 1;
                break;
            }
            case pt::DataType::kINT32: {
                int32_t* data_ptr = tensor_dense->mutable_data<int32_t>();
                for(int i = 0; i < tensor_dense->numel(); i++)
                    data_ptr[i] = 1;
                break;
            }
            case pt::DataType::kFLOAT64: {
                double* data_ptr = tensor_dense->mutable_data<double>();
                for(int i = 0; i < tensor_dense->numel(); i++)
                    data_ptr[i] = 1.0;
                break;
            }
            case pt::DataType::kFLOAT32: {
                float* data_ptr = tensor_dense->mutable_data<float>();
                for(int i = 0; i < tensor_dense->numel(); i++)
                    data_ptr[i] = 1.0;
                break;
            }
            default: {
                PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
                break;
            }
        }
        
        buffer[pos].SetImpl(tensor_dense);

    }

}

}  // namespace egr
