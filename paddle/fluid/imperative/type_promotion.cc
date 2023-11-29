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
#include "paddle/fluid/imperative/type_promotion.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"

namespace paddle {
namespace imperative {
TypePrmotionGuard::TypePrmotionGuard(std::shared_ptr<Tracer> tracer,
                                     bool use_type_promotion_)
    : tracer_(tracer) {
  pre_type_promotion = tracer_->UseTypePromotion();
  if (pre_type_promotion != use_type_promotion_) {
    tracer_->EnableTypePromotion();
    if (!use_type_promotion_) {
      tracer_->DisableTypePromotion();
    }
  }
}

TypePrmotionGuard::~TypePrmotionGuard() {
  if (pre_type_promotion) {
    tracer_->EnableTypePromotion();
  } else {
    tracer_->DisableTypePromotion();
  }
}
}  // namespace imperative
}  // namespace paddle
