// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/dynamic_shape_infermeta_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {

USE_TRT_DYNAMIC_INFER_META_FN(gather_nd);
USE_TRT_DYNAMIC_INFER_META_FN(yolo_box);
USE_TRT_DYNAMIC_INFER_META_FN(instance_norm);
USE_TRT_DYNAMIC_INFER_META_FN(unfold);
USE_TRT_DYNAMIC_INFER_META_FN(scatter_nd_add);
USE_TRT_DYNAMIC_INFER_META_FN(pad3d);
USE_TRT_DYNAMIC_INFER_META_FN(inverse);
USE_TRT_DYNAMIC_INFER_META_FN(grid_sampler);
USE_TRT_DYNAMIC_INFER_META_FN(fused_conv2d_add_act);
USE_TRT_DYNAMIC_INFER_META_FN(conv2d);
USE_TRT_DYNAMIC_INFER_META_FN(conv2d_transpose);
USE_TRT_DYNAMIC_INFER_META_FN(memory_efficient_attention);
USE_TRT_DYNAMIC_INFER_META_FN(p_norm);
USE_TRT_DYNAMIC_INFER_META_FN(pad);
USE_TRT_DYNAMIC_INFER_META_FN(argsort);
USE_TRT_DYNAMIC_INFER_META_FN(scatter);
USE_TRT_DYNAMIC_INFER_META_FN(solve);
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
