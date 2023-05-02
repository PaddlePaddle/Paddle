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

#include "paddle/phi/capi/include/c_kernel_factory.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/kernel_factory.h"

/**
 * TensorArgDef
 */

void PD_TensorArgDefSetDataLayout(PD_TensorArgDef* def,
                                  PD_DataLayout layout,
                                  PD_Status* status) {
  if (status) {
    if (!def) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_def = reinterpret_cast<phi::TensorArgDef*>(def);
  cc_def->SetDataLayout(phi::capi::ToPhiDataLayout(layout));
}

void PD_TensorArgDefSetDataType(PD_TensorArgDef* def,
                                PD_DataType dtype,
                                PD_Status* status) {
  if (status) {
    if (!def) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_def = reinterpret_cast<phi::TensorArgDef*>(def);
  cc_def->SetDataType(phi::capi::ToPhiDataType(dtype));
}

/**
 * KernelArgsDef
 */

PD_List PD_KernelArgsDefGetInputArgDefs(PD_KernelArgsDef* def,
                                        PD_Status* status) {
  PD_List list;
  if (status) {
    if (!def) {
      *status = C_FAILED;
      list.size = 0;
      list.data = nullptr;
      return list;
    }
    *status = C_SUCCESS;
  }
  auto cc_def = reinterpret_cast<phi::KernelArgsDef*>(def);
  auto& arg_defs = cc_def->input_defs();
  list.size = arg_defs.size();
  auto ptr = new PD_TensorArgDef*[list.size];
  list.data = ptr;
  for (size_t i = 0; i < list.size; ++i) {
    ptr[i] = reinterpret_cast<PD_TensorArgDef*>(&arg_defs[i]);
  }
  return list;
}

PD_List PD_KernelArgsDefGetOutputArgDefs(PD_KernelArgsDef* def,
                                         PD_Status* status) {
  PD_List list;
  if (status) {
    if (!def) {
      *status = C_FAILED;
      list.size = 0;
      list.data = nullptr;
      return list;
    }
    *status = C_SUCCESS;
  }
  auto cc_def = reinterpret_cast<phi::KernelArgsDef*>(def);
  auto& arg_defs = cc_def->output_defs();
  list.size = arg_defs.size();
  auto ptr = new PD_TensorArgDef*[list.size];
  list.data = ptr;
  for (size_t i = 0; i < list.size; ++i) {
    ptr[i] = reinterpret_cast<PD_TensorArgDef*>(&arg_defs[i]);
  }
  return list;
}

/**
 * KernelKey
 */

PD_DataLayout PD_KernelKeyGetLayout(PD_KernelKey* key, PD_Status* status) {
  if (status) {
    if (!key) {
      *status = C_FAILED;
      return PD_DataLayout::ALL_LAYOUT;
    }
    *status = C_SUCCESS;
  }
  auto cc_key = reinterpret_cast<phi::KernelKey*>(key);
  return phi::capi::ToPDDataLayout(cc_key->layout());
}

PD_DataType PD_KernelKeyGetDataType(PD_KernelKey* key, PD_Status* status) {
  if (status) {
    if (!key) {
      *status = C_FAILED;
      return PD_DataType::UNDEFINED;
    }
    *status = C_SUCCESS;
  }
  auto cc_key = reinterpret_cast<phi::KernelKey*>(key);
  return phi::capi::ToPDDataType(cc_key->dtype());
}

/**
 * Kernel
 */

PD_KernelArgsDef* PD_KernelGetArgsDef(PD_Kernel* kernel, PD_Status* status) {
  if (status) {
    if (!kernel) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }
  auto cc_kernel = reinterpret_cast<phi::Kernel*>(kernel);
  return reinterpret_cast<PD_KernelArgsDef*>(
      const_cast<phi::KernelArgsDef*>(&cc_kernel->args_def()));
}

PD_REGISTER_CAPI(kernel_factory);
