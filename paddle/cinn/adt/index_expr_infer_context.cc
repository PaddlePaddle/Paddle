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

#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/equation_function_constants_provider.h"

namespace cinn::adt {

Constant IndexExprInferContext::GetDimSize(const Dim& dim) const {
  return constants_provider_->GetDimSize(dim);
}

}  // namespace cinn::adt
