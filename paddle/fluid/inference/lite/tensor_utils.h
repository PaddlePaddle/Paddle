// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace inference {
namespace lite {
namespace utils {

template <typename DstTensor, typename SrcTensor>
void TensorCopyAsync(DstTensor* dst,
                     const SrcTensor& src,
                     const platform::DeviceContext& ctx);

template <typename DstTensor, typename SrcTensor>
void TensorDataShare(DstTensor* dst, SrcTensor* src);

template <typename DstTensor, typename SrcTensor>
void TensorCopy(DstTensor* dst,
                SrcTensor* src,
                const platform::DeviceContext& ctx,
                bool shared = true) {
  if (shared) {
    VLOG(3) << "TensorDataShare is running";
    TensorDataShare(dst, src);
  } else {
    VLOG(3) << "TensorCopyAsync is running";
    TensorCopyAsync(dst, *src, ctx);
  }
}

}  // namespace utils
}  // namespace lite
}  // namespace inference
}  // namespace paddle
