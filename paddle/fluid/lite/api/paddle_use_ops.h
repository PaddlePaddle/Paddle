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

// ATTENTION This can only include in a .cc file.

#include "paddle_lite_factory_helper.h"  // NOLINT

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(relu);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);
USE_LITE_OP(elementwise_add)
USE_LITE_OP(elementwise_sub)
USE_LITE_OP(square)
USE_LITE_OP(softmax)
USE_LITE_OP(dropout)
USE_LITE_OP(concat)
USE_LITE_OP(conv2d)
USE_LITE_OP(depthwise_conv2d)
USE_LITE_OP(pool2d)
USE_LITE_OP(batch_norm)
USE_LITE_OP(fusion_elementwise_sub_activation)
USE_LITE_OP(transpose)
USE_LITE_OP(transpose2)

USE_LITE_OP(fake_quantize_moving_average_abs_max);
USE_LITE_OP(fake_dequantize_max_abs);
USE_LITE_OP(calib);
