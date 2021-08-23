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

#include "paddle/fluid/eager/function_helper.h"

#include "paddle/top/api/all.h"
#include "paddle/top/core/dense_tensor.h"

template<typename T>
static void add_kernel(const pt::DenseTensor& t0, const pt::DenseTensor& t1, pt::DenseTensor& out) {
    const T* t0_ptr = t0.data<T>();
    const T* t1_ptr = t1.data<T>();
    T* out_ptr = out.mutable_data<T>();
    for(int i = 0; i < t0.numel(); i++) {
        out_ptr[i] = t0_ptr[i] + t1_ptr[i];
    }   
}

namespace egr {

void ScaleAPI(const pt::Tensor& x, float scale, float bias,
              bool bias_after_scale, std::vector<pt::Tensor>& outs) {

    // Run Forward Function
    auto dense_tensor = std::dynamic_pointer_cast<pt::DenseTensor>(x.impl());

    PADDLE_ENFORCE(outs.size() == 1,
            paddle::platform::errors::Fatal("ScaleAPI should only return 1 tensor"));
    PADDLE_ENFORCE(dense_tensor->backend() == pt::Backend::kCPU,
            paddle::platform::errors::Fatal("Only CPU Backend is supported for now"));
    
    // Init output tensor
    auto tensor_meta = std::make_unique<pt::TensorMeta>(dense_tensor->dims(), dense_tensor->backend(), 
          dense_tensor->type(), dense_tensor->layout());
    auto dense_out = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));

    auto dev_ctx = paddle::platform::CPUDeviceContext();
    switch(dense_tensor->type()) {
        case pt::DataType::kFLOAT64: {
            pt::Scale<double>(dev_ctx, *dense_tensor.get() /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kFLOAT32: {
            pt::Scale<float>(dev_ctx, *dense_tensor.get() /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kINT64: {
            pt::Scale<int64_t>(dev_ctx, *dense_tensor.get() /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        case pt::DataType::kINT32: {
            pt::Scale<int32_t>(dev_ctx, *dense_tensor.get() /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out.get()/* out tensor */);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Unsupported data type"));
            break;
        }
    }

    outs[0].SetImpl(dense_out);
}

void FillConstAPI(double value, const pt::DDim& ddim, const pt::Backend& backend, 
                  const pt::DataType& dtype, const pt::DataLayout& layout,
                  pt::Tensor& target) {

    // Create new tensor->impl and fill it with 1.0
    // Fill 1.0
    std::shared_ptr<pt::DenseTensor> tensor_dense = nullptr;
    if(!target.defined() || !target.initialized()) {
        std::unique_ptr<pt::TensorMeta> tensor_meta = std::make_unique<pt::TensorMeta>(ddim, backend, dtype, layout);
        tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
        target.SetImpl(tensor_dense);

    } else {
        tensor_dense = std::dynamic_pointer_cast<pt::DenseTensor>(target.impl());
    }

    PADDLE_ENFORCE(tensor_dense != nullptr, 
        paddle::platform::errors::Fatal("FillConstAPI Only supports InputBuffer with DenseTensor for now."));
    PADDLE_ENFORCE(tensor_dense->backend() == pt::Backend::kCPU, 
        paddle::platform::errors::Fatal("FillConstAPI Only supports tensors with CPU backend for now."));
    
    switch(tensor_dense->type()) {
        case pt::DataType::kINT64: {
            int64_t* data_ptr = tensor_dense->mutable_data<int64_t>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<int64_t>(value);
            break;
        }
        case pt::DataType::kINT32: {
            int32_t* data_ptr = tensor_dense->mutable_data<int32_t>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<int32_t>(value);
            break;
        }
        case pt::DataType::kFLOAT64: {
            double* data_ptr = tensor_dense->mutable_data<double>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<double>(value);
            break;
        }
        case pt::DataType::kFLOAT32: {
            float* data_ptr = tensor_dense->mutable_data<float>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<float>(value);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
            break;
        }
    }
}

void AccumulateTensorsAPI(pt::Tensor& t0, const pt::Tensor& t1) {
    // Accumulate to t0
    std::shared_ptr<pt::DenseTensor> t0_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t0.impl());
    std::shared_ptr<pt::DenseTensor> t1_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t1.impl());
    
    PADDLE_ENFORCE(t0_dense != nullptr, 
        paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports InputBuffer with DenseTensor for now."));
    PADDLE_ENFORCE(t0_dense->backend() == pt::Backend::kCPU, 
        paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports tensors with CPU backend for now."));
    PADDLE_ENFORCE(t0.initialized(), 
        paddle::platform::errors::Fatal("Tensors to accumulate has not been initialized"));
    
    PADDLE_ENFORCE(t1_dense != nullptr, 
        paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports InputBuffer with DenseTensor for now."));
    PADDLE_ENFORCE(t1_dense->backend() == pt::Backend::kCPU, 
        paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports tensors with CPU backend for now."));
    PADDLE_ENFORCE(t1.initialized(), 
        paddle::platform::errors::Fatal("Tensors to accumulate has not been initialized"));
    
    PADDLE_ENFORCE(t1.type() == t0.type(), 
        paddle::platform::errors::Fatal("Unable to accumulate tensors with different dtype"));
    PADDLE_ENFORCE(t1.numel() == t0.numel(), 
        paddle::platform::errors::Fatal("Unable to accumulate tensors with different sizes"));
    

    // TODO: Replace this with call to add_kernel_api
    switch(t0.type()) {
        case pt::DataType::kINT64: {
            add_kernel<int64_t>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kINT32: {
            add_kernel<int32_t>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kFLOAT64: {
            add_kernel<double>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kFLOAT32: {
            add_kernel<float>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
            break;
        }
    }
}

} // namespace egr
