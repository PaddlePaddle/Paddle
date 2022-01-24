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

#include "paddle/pten/backends/dynload/mklrt.h"

namespace pten {
namespace dynload {

std::once_flag mklrt_dso_flag;
void* mklrt_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MKLDFTI_ROUTINE_EACH(DEFINE_WRAP);

DFTI_EXTERN MKL_LONG DftiCreateDescriptorX(DFTI_DESCRIPTOR_HANDLE* desc,
                                           enum DFTI_CONFIG_VALUE prec,
                                           enum DFTI_CONFIG_VALUE domain,
                                           MKL_LONG dim,
                                           MKL_LONG* sizes) {
  if (prec == DFTI_SINGLE) {
    if (dim == 1) {
      return DftiCreateDescriptor_s_1d(desc, domain, sizes[0]);
    } else {
      return DftiCreateDescriptor_s_md(desc, domain, dim, sizes);
    }
  } else if (prec == DFTI_DOUBLE) {
    if (dim == 1) {
      return DftiCreateDescriptor_d_1d(desc, domain, sizes[0]);
    } else {
      return DftiCreateDescriptor_d_md(desc, domain, dim, sizes);
    }
  } else {
    return DftiCreateDescriptor(desc, prec, domain, dim, sizes);
  }
}

}  // namespace dynload
}  // namespace pten
