// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/trait/custom_vjp.h"
#include "paddle/fluid/pir/dialect/operator/trait/forward_only.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#endif
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InplaceTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CustomVjpTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ForwardOnlyTrait)

#ifdef PADDLE_WITH_DNNL
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNOnlyTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNDynamicFallbackTrait)
#endif
