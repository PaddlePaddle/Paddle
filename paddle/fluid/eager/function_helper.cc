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

} // namespace egr
