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

#include "paddle/infrt/dialect/init_dialects.h"

#include <glog/logging.h>

#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/infrt/ir/basic_kernels.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"

#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/dialect/phi/ir/phi_kernels.h"

#include "paddle/infrt/dialect/tensor_shape.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
void registerCinnDialects(mlir::DialectRegistry &registry) {  // NOLINT
  registry.insert<ts::TensorShapeDialect,
                  InfrtDialect,
                  dt::DTDialect,
                  pd::PaddleDialect,
                  trt::TensorRTDialect
#ifdef INFRT_WITH_PHI
                  ,
                  phi::PHIDenseTensorDialect,
                  phi::PHICPUKernelDialect,
                  phi::PHIGPUKernelDialect,
                  phi::PHIDialect
#endif
                  >();
}
}  // namespace infrt
