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

#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/storage_properties.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#endif
#include "paddle/phi/core/utils/type_info.h"

namespace phi {

template <typename BaseT, typename DerivedT>
TypeInfoTraits<BaseT, DerivedT>::TypeInfoTraits() {
  static_cast<BaseT*>(static_cast<DerivedT*>(this))->type_info_ = kType;
}

template <typename BaseT, typename DerivedT>
bool TypeInfoTraits<BaseT, DerivedT>::classof(const BaseT* obj) {
  return obj->type_info() == kType;
}

template <typename BaseT, typename DerivedT>
const TypeInfo<BaseT> TypeInfoTraits<BaseT, DerivedT>::kType =
    RegisterStaticType<BaseT>(DerivedT::name());

template class TypeInfoTraits<phi::TensorBase, DenseTensor>;
template class TypeInfoTraits<phi::TensorBase, SelectedRows>;
template class TypeInfoTraits<phi::TensorBase, SparseCooTensor>;
template class TypeInfoTraits<phi::TensorBase, SparseCsrTensor>;
template class TypeInfoTraits<phi::TensorBase, StringTensor>;
template class TypeInfoTraits<phi::TensorBase, TensorArray>;

template class TypeInfoTraits<phi::DeviceContext, CPUContext>;
template class TypeInfoTraits<phi::DeviceContext, CustomContext>;

#ifdef PADDLE_WITH_DISTRIBUTE
template class TypeInfoTraits<phi::TensorBase, phi::distributed::DistTensor>;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU_KP)
template class TypeInfoTraits<phi::DeviceContext, GPUContext>;
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template class TypeInfoTraits<phi::DeviceContext, GPUPinnedContext>;
#endif

#ifdef PADDLE_WITH_XPU
template class TypeInfoTraits<phi::DeviceContext, XPUContext>;
#endif

#ifdef PADDLE_WITH_DNNL
template class TypeInfoTraits<phi::StorageProperties, OneDNNStorageProperties>;
#endif

template class TypeInfoTraits<phi::StorageProperties, NPUStorageProperties>;

}  // namespace phi
