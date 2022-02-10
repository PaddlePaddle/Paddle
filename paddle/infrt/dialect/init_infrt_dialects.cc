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

#include "paddle/infrt/dialect/init_infrt_dialects.h"

#include <glog/logging.h>

#include "paddle/infrt/dialect/basic_kernels.h"
#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/pd_ops.h"
#include "paddle/infrt/dialect/pten/infrt_pten_tensor.h"
#include "paddle/infrt/dialect/pten/pten_base.h"
#include "paddle/infrt/dialect/tensor_shape.h"

namespace infrt {
void registerCinnDialects(mlir::DialectRegistry &registry) {  // NOLINT
  registry.insert<ts::TensorShapeDialect,
                  dialect::INFRTDialect,
                  dt::DTDialect,
                  mlir::pd::PaddleDialect,
#ifdef INFRT_WITH_PTEN
                  pten::PTENDenseTensorDialect,
                  pten::PTENDialect
#endif
                  >();
}
}  // namespace infrt
