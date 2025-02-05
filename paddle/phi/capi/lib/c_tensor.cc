// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/capi/include/c_tensor.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

PD_DataType PD_TensorGetPDDataType(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataType::UNDEFINED;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return phi::capi::ToPDDataType(cc_tensor->dtype());
}

PD_DataLayout PD_TensorGetDataLayout(const PD_Tensor* tensor,
                                     PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataLayout::ALL_LAYOUT;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return phi::capi::ToPDDataLayout(cc_tensor->layout());
}

int64_t PD_TensorGetByteSize(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->memory_size();
}

void* PD_TensorGetDataPointer(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return const_cast<void*>(cc_tensor->data());
}

int64_t PD_TensorGetElementCount(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->numel();
}

int64_t PD_TensorGetNumDims(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->dims().size();
}

int64_t PD_TensorGetDim(const PD_Tensor* tensor,
                        size_t index,
                        PD_Status* status) {
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);

  if (status) {
    if (!tensor || index >= static_cast<size_t>(cc_tensor->dims().size())) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  return cc_tensor->dims()[index];
}

int64_t PD_TensorGetNumStrides(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->strides().size();
}

int64_t PD_TensorGetStride(const PD_Tensor* tensor,
                           size_t index,
                           PD_Status* status) {
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);

  if (status) {
    if (!tensor || index >= static_cast<size_t>(cc_tensor->strides().size())) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  return cc_tensor->strides()[index];
}

void PD_TensorGetLoD(const PD_Tensor* tensor,
                     PD_List* data,
                     PD_List* offset,
                     PD_Status* status) {
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);

  if (status) {
    if (!tensor || !data || !offset) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto lod = cc_tensor->lod();
  offset->size = lod.size() + 1;
  auto offset_data = new size_t[offset->size];
  offset->data = offset_data;
  offset_data[0] = 0;

  size_t sz = 0;
  for (size_t i = 0; i < lod.size(); ++i) {
    offset_data[i + 1] = lod[i].size() + offset_data[i];
    sz += lod[i].size();
  }

  auto data_ptr = new size_t[sz];
  data->data = data_ptr;
  data->size = sz;
  for (size_t i = 0; i < lod.size(); ++i) {
    memcpy(data_ptr, lod[i].data(), lod[i].size() * sizeof(size_t));
    data_ptr += lod[i].size();
  }
}

bool PD_TensorIsInitialized(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return false;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->initialized();
}

bool PD_TensorIsValid(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return false;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->valid();
}

void* PD_TensorGetHolder(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->Holder().get();
}

size_t PD_TensorGetOffset(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->offset();
}

void PD_TensorSetDims(PD_Tensor* tensor,
                      int64_t ndims,
                      const int64_t* dims,
                      PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  std::vector<int> shape(dims, dims + ndims);
  cc_tensor->Resize(common::make_ddim(shape));
}

void PD_TensorSetOffset(PD_Tensor* tensor,
                        const int64_t offset,
                        PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->set_offset(offset);
}

void PD_TensorSetStrides(PD_Tensor* tensor,
                         int64_t nstrides,
                         const int64_t* strides,
                         PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  std::vector<int> shape(strides, strides + nstrides);
  cc_tensor->set_strides(common::make_ddim(shape));
}

void PD_TensorSetDataType(PD_Tensor* tensor,
                          PD_DataType dtype,
                          PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->set_type(phi::capi::ToPhiDataType(dtype));
}

void PD_TensorSetDataLayout(PD_Tensor* tensor,
                            PD_DataLayout layout,
                            PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->set_layout(phi::capi::ToPhiDataLayout(layout));
}

void PD_TensorResetLoD(PD_Tensor* tensor,
                       PD_List data,
                       PD_List offset,
                       PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  phi::LegacyLoD lod;
  auto offset_ptr = static_cast<size_t*>(offset.data);
  auto data_ptr = static_cast<size_t*>(data.data);

  for (size_t i = 0; i < offset.size - 1; ++i) {
    lod.emplace_back(data_ptr + offset_ptr[i], data_ptr + offset_ptr[i + 1]);
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->ResetLoD(lod);
}

PD_Tensor* PD_NewTensor() {
  return reinterpret_cast<PD_Tensor*>(new phi::DenseTensor());
}

void PD_DeleteTensor(PD_Tensor* tensor) {
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  delete cc_tensor;
}

void PD_TensorShareDataWith(PD_Tensor* dst,
                            const PD_Tensor* src,
                            PD_Status* status) {
  if (status) {
    if (!dst || !src) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_dst_tensor = reinterpret_cast<phi::DenseTensor*>(dst);
  auto cc_src_tensor = reinterpret_cast<const phi::DenseTensor*>(src);
  cc_dst_tensor->ShareDataWith(*cc_src_tensor);
}

void PD_TensorShareLoDWith(PD_Tensor* dst,
                           const PD_Tensor* src,
                           PD_Status* status) {
  if (status) {
    if (!dst || !src) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_dst_tensor = reinterpret_cast<phi::DenseTensor*>(dst);
  auto cc_src_tensor = const_cast<phi::DenseTensor*>(
      reinterpret_cast<const phi::DenseTensor*>(src));

  phi::MetaTensor meta_dst(cc_dst_tensor);
  const phi::MetaTensor meta_src(cc_src_tensor);
  meta_dst.share_lod(meta_src);
}

PD_Tensor* PD_OptionalTensorGetPointer(PD_Tensor* tensor) {
  auto cc_tensor =
      reinterpret_cast<paddle::optional<phi::DenseTensor>*>(tensor);
  return reinterpret_cast<PD_Tensor*>(cc_tensor->get_ptr());
}

PD_List PD_TensorVectorToList(PD_Tensor* tensor) {
  auto cc_tensor =
      reinterpret_cast<std::vector<const phi::DenseTensor*>*>(tensor);
  PD_List list;
  list.size = cc_tensor->size();
  list.data = cc_tensor->data();
  return list;
}

PD_REGISTER_CAPI(tensor);
