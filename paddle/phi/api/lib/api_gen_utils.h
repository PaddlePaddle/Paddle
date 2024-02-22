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
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
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

paddle::optional<phi::DenseTensor> TensorToDenseTensor(
    const paddle::optional<Tensor>& tensor);

std::unique_ptr<std::vector<phi::DenseTensor*>> TensorToDenseTensor(
    const std::vector<Tensor>& tensors);

std::vector<const phi::DenseTensor*> TensorToConstDenseTensorPtr(
    const std::vector<Tensor>& tensors);

paddle::optional<std::vector<const phi::DenseTensor*>>
TensorToConstDenseTensorPtr(
    const paddle::optional<std::vector<Tensor>>& tensors);

std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(const Tensor& tensor);

paddle::optional<phi::SelectedRows> TensorToSelectedRows(
    const paddle::optional<Tensor>& tensor);

std::shared_ptr<phi::StringTensor> TensorToStringTensor(const Tensor& tensor);

std::shared_ptr<phi::SparseCooTensor> TensorToSparseCooTensor(
    const Tensor& tensor);
/* ----------------- for infer_meta --------------------- */

phi::MetaTensor MakeMetaTensor(const phi::TensorBase& tensor);

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::DenseTensor>& tensor);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::DenseTensor*>& tensors);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<phi::DenseTensor*>& tensors);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::SelectedRows*>& tensors);

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SelectedRows>& tensor);

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SparseCooTensor>& tensor);

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SparseCsrTensor>& tensor);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<std::vector<const phi::DenseTensor*>>& tensors);

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::TensorBase*>& tensors);

/* ------------------ for output ----------------------- */

phi::DenseTensor* SetKernelOutput(Tensor* out);

std::vector<phi::DenseTensor*> SetKernelOutput(size_t out_size,
                                               std::vector<Tensor>* out);

std::vector<phi::DenseTensor*> SetInplaceVectorKernelOutput(
    size_t out_size, std::vector<Tensor>* out);

std::vector<phi::DenseTensor*> SetInplaceOptionalVectorKernelOutput(
    size_t out_size, const paddle::optional<std::vector<Tensor>>& out);

// For backward api
std::vector<phi::DenseTensor*> SetKernelOutput(std::vector<Tensor*>* out);

phi::SelectedRows* SetSelectedRowsKernelOutput(Tensor* out);

phi::TensorBase* SetSparseKernelOutput(Tensor* out, TensorType type);

phi::TensorBase* SetStringsKernelOutput(Tensor* out, TensorType type);

phi::DenseTensor* ProcessStrideBackup(phi::DenseTensor** tensor);

std::vector<phi::DenseTensor*> ProcessStrideBackup(
    std::vector<phi::DenseTensor*>* tensor);

phi::SelectedRows* ProcessStrideBackup(phi::SelectedRows** tensor);

void TransStride(phi::DeviceContext* dev_ctx,
                 phi::DenseTensor* from,
                 phi::DenseTensor* to);

void TransStride(phi::DeviceContext* dev_ctx,
                 const std::vector<phi::DenseTensor*>& from,
                 const std::vector<phi::DenseTensor*>& to);

void TransStride(phi::DeviceContext* dev_ctx,
                 phi::SelectedRows* from,
                 phi::SelectedRows* to);

void TransStrideLegacy(phi::DeviceContext* dev_ctx,
                       phi::DenseTensor* from,
                       phi::DenseTensor* to);

/* ------------------ for auto parallel ----------------------- */

phi::distributed::DistMetaTensor MakeDistMetaTensor(
    const phi::TensorBase& tensor);

std::vector<phi::distributed::DistMetaTensor> MakeDistMetaTensor(
    const std::vector<const phi::TensorBase*>& tensors);

phi::distributed::DistTensor* SetKernelDistOutput(
    Tensor* out,
    const phi::distributed::ArgDistAttr& dist_attr =
        phi::distributed::TensorDistAttr());

// For backward
std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    std::vector<Tensor*> out);

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    size_t out_size, std::vector<Tensor>* out);

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    const phi::distributed::ArgDistAttr& dist_attr, std::vector<Tensor>* out);

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out,
    bool set_dist_output_as_tensor_impl,
    const phi::distributed::TensorDistAttr& dist_attr);

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out,
    bool set_dist_output_as_tensor_impl,
    const phi::distributed::ArgDistAttr& dist_attr =
        phi::distributed::TensorDistAttr());

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
CreateKernelDistOutput(std::vector<Tensor*> out,
                       bool set_dist_output_as_tensor_impl,
                       const phi::distributed::ArgDistAttr& dist_attr);

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
CreateKernelDistOutput(std::vector<Tensor*> out,
                       bool set_dist_output_as_tensor_impl);

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out, const phi::distributed::ArgDistAttr& dist_attr);

// DistTensor need to set initial dist attr after the dims setted, it is
// constructed based dims and current process mesh, before calling this
// function, the out should hold correct dims
void SetReplicatedDistAttrForOutput(
    phi::distributed::DistTensor* out,
    const phi::distributed::ProcessMesh& process_mesh);

}  // namespace experimental
}  // namespace paddle
