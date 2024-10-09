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

#include "paddle/phi/capi/include/c_device_context.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/dense_tensor.h"

PD_Stream PD_DeviceContextGetStream(const PD_DeviceContext* ctx,
                                    PD_Status* status) {
  if (status) {
    if (!ctx) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }
  auto dev_ctx_type =
      reinterpret_cast<const phi::CustomContext*>(ctx)->GetPlace().GetType();
  if (dev_ctx_type == phi::AllocationType::CUSTOM) {
    return reinterpret_cast<PD_Stream>(
        reinterpret_cast<const phi::CustomContext*>(ctx)->stream());
  } else if (dev_ctx_type == phi::AllocationType::CPU) {
    return nullptr;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (dev_ctx_type == phi::AllocationType::GPU) {
    return reinterpret_cast<PD_Stream>(
        reinterpret_cast<const phi::GPUContext*>(ctx)->stream());
#endif
#ifdef PADDLE_WITH_XPU
  } else if (dev_ctx_type == phi::AllocationType::XPU) {
    return nullptr;
#endif
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Only support Custom/CPU/GPU/XPU DeviceContext"));
  }
}

void* PD_DeviceContextAllocateTensor(const PD_DeviceContext* ctx,
                                     PD_Tensor* tensor,
                                     size_t size,
                                     PD_DataType dtype,
                                     PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }

  auto dev_ctx = reinterpret_cast<const phi::DeviceContext*>(ctx);
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  auto phi_dtype = phi::capi::ToPhiDataType(dtype);
  if (ctx) {
    return dev_ctx->Alloc(cc_tensor, phi_dtype, size);
  } else {
    auto place = phi::CPUPlace();
    return cc_tensor->mutable_data(place, phi_dtype, size);
  }
}

void PD_DeviceContextSetSeed(const PD_DeviceContext* ctx,
                             uint64_t seed,
                             PD_Status* status) {
  if (status) {
    *status = C_SUCCESS;
  }
  auto dev_ctx = reinterpret_cast<const phi::DeviceContext*>(ctx);
  dev_ctx->GetGenerator()->SetCurrentSeed(seed);
}

uint64_t PD_DeviceContextGetSeed(const PD_DeviceContext* ctx,
                                 PD_Status* status) {
  if (status) {
    *status = C_SUCCESS;
  }
  auto dev_ctx = reinterpret_cast<const phi::DeviceContext*>(ctx);
  return dev_ctx->GetGenerator()->GetCurrentSeed();
}

uint64_t PD_DeviceContextGetRandom(const PD_DeviceContext* ctx,
                                   PD_Status* status) {
  if (status) {
    *status = C_SUCCESS;
  }
  auto dev_ctx = reinterpret_cast<const phi::DeviceContext*>(ctx);
  return dev_ctx->GetGenerator()->Random64();
}

PD_REGISTER_CAPI(device_context);
