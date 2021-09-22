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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/tcmpt/api/include/dev/core.h"

namespace paddle {
namespace framework {

template <typename PtTensorImplT, typename VariableT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const VariableT& tensor,
                                              pt::Backend backend,
                                              pt::DataType dtype,
                                              pt::DataLayout layout);

template <typename PtTensorImplT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const LoDTensor& tensor,
                                              const platform::Place& place,
                                              proto::VarType::Type type);

template <typename PtTensorImplT>
std::shared_ptr<PtTensorImplT> MakeTensorImpl(const Tensor& tensor,
                                              const platform::Place& place,
                                              proto::VarType::Type type);

template <typename PtTensorImplT>
void ShareTensorImpl(PtTensorImplT* tensor_impl, LoDTensor* out);

template <typename PtTensorImplT>
void ShareTensorImpl(PtTensorImplT* tensor_impl, Tensor* out);

}  // namespace framework
}  // namespace paddle
