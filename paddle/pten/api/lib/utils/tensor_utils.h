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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_factory.h"

namespace paddle {
namespace experimental {

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::Tensor& src);

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::LoDTensor& src);

pten::Scalar MakePtenScalar(const paddle::framework::LoDTensor& src);

pten::ScalarArray MakePtenScalarArray(const paddle::framework::LoDTensor& src);

pten::Scalar MakePtenScalarFromVar(const framework::Variable& variable);

pten::ScalarArray MakePtenScalarArrayFromVar(
    const framework::Variable& variable);

pten::ScalarArray MakePtenScalarArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list);

std::unique_ptr<pten::TensorBase> MakePtenTensorBaseFromVar(
    const framework::Variable& variable, const pten::TensorArgDef& arg_def);

std::unique_ptr<pten::TensorBase> MakePtenTensorBaseFromVar(
    framework::Variable* variable, const pten::TensorArgDef& arg_def);

void MovesStorage(pten::DenseTensor* src, paddle::framework::Tensor* dst);

void MovesStorage(pten::DenseTensor* src, paddle::framework::LoDTensor* dst);

void MovesSharedStorage(pten::DenseTensor* src, paddle::framework::Tensor* dst);

void MovesSharedStorage(pten::DenseTensor* src,
                        paddle::framework::LoDTensor* dst);

/**
 * In order to improve the compatibility state performance, some tricky tool
 * functions are added.
 *
 * The ReMake** function takes out the LoDTensor information and directly
 * replaces it with the corresponding member of the DenseTensor to avoid
 * the overhead caused by frequent construction and destruction of the
 * DenseTensor.
 */

void ReMakePtenDenseTensor(const paddle::framework::Tensor& src,
                           const pten::TensorArgDef& arg_def,
                           pten::DenseTensor* dst);

void ReMakePtenDenseTensor(const paddle::framework::LoDTensor& src,
                           const pten::TensorArgDef& arg_def,
                           pten::DenseTensor* dst);

void ReMakePtenDenseTensorFromVar(const framework::Variable& variable,
                                  const pten::TensorArgDef& arg_def,
                                  pten::DenseTensor* dst);

void ReMakePtenDenseTensorFromVar(framework::Variable* variable,
                                  const pten::TensorArgDef& arg_def,
                                  pten::DenseTensor* dst);

void MakeVariableFromPtenTensor(pten::DenseTensor* src,
                                framework::Variable* variable);

}  // namespace experimental
}  // namespace paddle
