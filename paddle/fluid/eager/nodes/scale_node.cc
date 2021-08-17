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

#include "paddle/fluid/eager/nodes/scale_node.h"
#include "paddle/top/api/all.h"

#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

void GradNodeScale::SetAttributes(float scale) {
    scale_ = scale;
}

std::vector<pt::Tensor> GradNodeScale::operator()(const std::vector<pt::Tensor>& grads) {
    PADDLE_ENFORCE(grads.size() == 1,
                paddle::platform::errors::Fatal("ScaleGradNode should take exactly 1 grad tensor"
                                                "However received: %d", grads.size()));
    
    // Apply Gradient Hooks
    std::shared_ptr<pt::DenseTensor> dense_grad(nullptr);
    if(GradientHooksRegistered()) {
        std::vector<pt::Tensor> hooked_grads = ApplyGradientHooks(grads);
        dense_grad = std::dynamic_pointer_cast<pt::DenseTensor>(hooked_grads[0].impl());
    } else {
        dense_grad = std::dynamic_pointer_cast<pt::DenseTensor>(grads[0].impl());
    }
    
    // Apply Reduce Hooks
    if(ReduceHooksRegistered()) {
        ApplyReduceHooks();
    }

    // Handle input tensor
    PADDLE_ENFORCE(dense_grad != nullptr,
                paddle::platform::errors::Fatal("Only DenseTensor is supported for now"));
    PADDLE_ENFORCE(dense_grad->backend() == pt::Backend::kCPU,
                paddle::platform::errors::Fatal("Only CPU Backend is supported for now"));
    
    // Init output tensor
    auto tensor_meta = std::make_unique<pt::TensorMeta>(dense_grad->dims(), dense_grad->backend(), 
          dense_grad->type(), dense_grad->layout());
    auto dense_out = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));

    auto dev_ctx = paddle::platform::CPUDeviceContext();
    switch(dense_grad->type()) {
        case pt::DataType::kFLOAT64: {
            pt::Scale<double>(dev_ctx, *dense_grad.get() /* grad tensor */, scale_ /* scale */, 0.0/* bias */, true/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kFLOAT32: {
            pt::Scale<float>(dev_ctx, *dense_grad.get() /* grad tensor */, scale_ /* scale */, 0.0/* bias */, true/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kINT64: {
            pt::Scale<int64_t>(dev_ctx, *dense_grad.get() /* grad tensor */, scale_ /* scale */, 0.0/* bias */, true/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kINT32: {
            pt::Scale<int32_t>(dev_ctx, *dense_grad.get() /* grad tensor */, scale_ /* scale */, 0.0/* bias */, true/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Unsupported data type"));
            break;
        }
    }
    auto out_impl = std::dynamic_pointer_cast<pt::TensorInterface>(dense_out);
    auto out_tensor = pt::Tensor(out_impl);
    std::vector<pt::Tensor> out = { std::move(out_tensor) };
        
    return out;
}

} // namespace egr
