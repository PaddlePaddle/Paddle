/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/string_tensor.h"

namespace paddle {
namespace experimental {

enum class TensorType { DENSE_TENSOR, SPARSE_CSR, SPARSE_COO, STRING_TENSOR };

/* ------------------ for input ----------------------- */

std::shared_ptr<phi::DenseTensor> TensorToDenseTensor(const Tensor& tensor);

std::shared_ptr<phi::DenseTensor> TensorToDenseTensor(
    const paddle::optional<Tensor>& tensor);

std::unique_ptr<std::vector<phi::DenseTensor>> TensorToDenseTensor(
    const std::vector<Tensor>& tensors);

std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(const Tensor& tensor);

std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(
    const paddle::optional<const Tensor&>& tensor);

std::shared_ptr<phi::StringTensor> TensorToStringTensor(const Tensor& tensor);

/* ----------------- for infer_meta --------------------- */

phi::MetaTensor MakeMetaTensor(const phi::DenseTensor& tensor);

paddle::optional<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<const phi::DenseTensor&>& tensor);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::DenseTensor*>& tensors);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<phi::DenseTensor*>& tensors);

phi::MetaTensor MakeMetaTensor(const phi::SelectedRows& tensor);

paddle::optional<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<const phi::SelectedRows&>& tensor);

phi::MetaTensor MakeMetaTensor(const phi::StringTensor& tensor);

/* ------------------ for output ----------------------- */

phi::DenseTensor* SetKernelOutput(Backend backend, Tensor* out);

std::vector<phi::DenseTensor*> SetKernelOutput(size_t out_size,
                                               Backend backend,
                                               std::vector<Tensor>* out);

phi::SelectedRows* SetSelectedRowsKernelOutput(Backend backend, Tensor* out);

phi::TensorBase* SetSparseKernelOutput(Tensor* out, TensorType type);

phi::TensorBase* SetStringsKernelOutput(Backend backend,
                                        Tensor* out,
                                        TensorType type);

}  // namespace experimental
}  // namespace paddle
