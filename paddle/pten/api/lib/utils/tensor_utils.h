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

#include <memory>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/compat/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_factory.h"

namespace paddle {
namespace experimental {

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::Tensor& src);

pten::ScalarArray MakePtenScalarArray(const paddle::framework::Tensor& src);

pten::Scalar MakePtenScalarFromVar(const framework::Variable& variable);

pten::ScalarArray MakePtenScalarArrayFromVar(
    const framework::Variable& variable);

pten::ScalarArray MakePtenScalarArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list);

void ResetTensorDtypeAndLayoutByArgDef(pten::TensorBase* dst,
                                       const pten::TensorArgDef& arg_def);

}  // namespace experimental
}  // namespace paddle
