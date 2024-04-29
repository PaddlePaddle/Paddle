/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <popart/names.hpp>

namespace paddle {
namespace platform {
namespace ipu {

static constexpr const char *sIpuIndexAttr = "ipu_index";
static constexpr const char *sIpuStageAttr = "ipu_stage";
static constexpr const char *sMatmulSerializeFactor = "serialize_factor";
static constexpr const char *sMatmulSerializeMode = "serialize_mode";
static constexpr const char *sAvailMemAttribute = "__available_memory";
static constexpr const char *sOpNamescope = "op_namescope";
static constexpr const char *sOpIdentifyIdAttr = "op_identify_id";
static constexpr const char *sDebugInfoId = "__debug_info_id";

static constexpr const char *sBeta1 = "beta1";
static constexpr const char *sBeta2 = "beta2";
static constexpr const char *sBeta1Pow = "Beta1Pow";
static constexpr const char *sBeta2Pow = "Beta2Pow";
static constexpr const char *sLossScaling = "LossScaling";

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
