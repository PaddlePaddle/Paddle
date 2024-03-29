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

#include "paddle/phi/capi/include/c_meta_tensor.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/meta_tensor.h"

PD_DataType PD_MetaTensorGetPDDataType(const PD_MetaTensor *tensor,
                                       PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataType::UNDEFINED;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return phi::capi::ToPDDataType(cc_tensor->dtype());
}

PD_DataLayout PD_MetaTensorGetDataLayout(const PD_MetaTensor *tensor,
                                         PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataLayout::ALL_LAYOUT;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return phi::capi::ToPDDataLayout(cc_tensor->layout());
}

int64_t PD_MetaTensorGetElementCount(const PD_MetaTensor *tensor,
                                     PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return cc_tensor->numel();
}

int64_t PD_MetaTensorGetNumDims(const PD_MetaTensor *tensor,
                                PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return cc_tensor->dims().size();
}

int64_t PD_MetaTensorGetDim(const PD_MetaTensor *tensor,
                            size_t index,
                            PD_Status *status) {
  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);

  if (status) {
    if (!tensor || index >= static_cast<size_t>(cc_tensor->dims().size())) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  return cc_tensor->dims()[index];
}

int64_t PD_MetaTensorGetNumStrides(const PD_MetaTensor *tensor,
                                   PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return cc_tensor->strides().size();
}

int64_t PD_MetaTensorGetStride(const PD_MetaTensor *tensor,
                               size_t index,
                               PD_Status *status) {
  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);

  if (status) {
    if (!tensor || index >= static_cast<size_t>(cc_tensor->strides().size())) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  return cc_tensor->strides()[index];
}

bool PD_MetaTensorIsValid(const PD_MetaTensor *tensor, PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return false;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::MetaTensor *>(tensor);
  return cc_tensor->initialized();
}

void PD_MetaTensorSetDims(PD_MetaTensor *tensor,
                          int64_t ndims,
                          const int64_t *dims,
                          PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::MetaTensor *>(tensor);
  std::vector<int> shape(dims, dims + ndims);
  cc_tensor->set_dims(common::make_ddim(shape));
}

void PD_MetaTensorSetStrides(PD_MetaTensor *tensor,
                             int64_t nstrides,
                             const int64_t *strides,
                             PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::MetaTensor *>(tensor);
  std::vector<int> shape(strides, strides + nstrides);
  cc_tensor->set_strides(common::make_ddim(shape));
}

void PD_MetaTensorSetDataType(PD_MetaTensor *tensor,
                              PD_DataType dtype,
                              PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::MetaTensor *>(tensor);
  cc_tensor->set_dtype(phi::capi::ToPhiDataType(dtype));
}

void PD_MetaTensorSetDataLayout(PD_MetaTensor *tensor,
                                PD_DataLayout layout,
                                PD_Status *status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::MetaTensor *>(tensor);
  cc_tensor->set_layout(phi::capi::ToPhiDataLayout(layout));
}

PD_REGISTER_CAPI(meta_tensor);
