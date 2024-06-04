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

#include "paddle/fluid/prim/utils/static/static_global_utils.h"

namespace paddle::prim {
StaticCompositeContext* StaticCompositeContext::static_composite_context_ =
    new StaticCompositeContext();
thread_local bool StaticCompositeContext::enable_bwd_prim_ = false;
thread_local bool StaticCompositeContext::enable_fwd_prim_ = false;
thread_local bool StaticCompositeContext::enable_eager_prim_ = false;
}  // namespace paddle::prim
