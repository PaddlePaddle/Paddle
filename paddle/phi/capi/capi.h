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

#if !defined(_WIN32)

#include "paddle/phi/capi/include/common.h"

PD_DECLARE_CAPI(data_type);
PD_DECLARE_CAPI(device_context);
PD_DECLARE_CAPI(int_array);
PD_DECLARE_CAPI(kernel_context);
PD_DECLARE_CAPI(kernel_factory);
PD_DECLARE_CAPI(kernel_registry);
PD_DECLARE_CAPI(place);
PD_DECLARE_CAPI(scalar);
PD_DECLARE_CAPI(tensor);

#endif
