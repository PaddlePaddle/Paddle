/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/storage_properties.h"

namespace phi {

std::unique_ptr<StorageProperties> CopyStorageProperties(
    const std::unique_ptr<StorageProperties>& sp) {
  if (sp) {
    if (NPUStorageProperties::classof(sp.get())) {
      auto result = std::make_unique<NPUStorageProperties>();
      result->storage_format =
          static_cast<NPUStorageProperties*>(sp.get())->storage_format;
      result->storage_dims =
          static_cast<NPUStorageProperties*>(sp.get())->storage_dims;
      return result;
#ifdef PADDLE_WITH_DNNL
    } else if (OneDNNStorageProperties::classof(sp.get())) {
      auto result = std::make_unique<OneDNNStorageProperties>();
      result->format = static_cast<OneDNNStorageProperties*>(sp.get())->format;
      result->mem_desc =
          static_cast<OneDNNStorageProperties*>(sp.get())->mem_desc;
      return result;
#endif
#ifdef PADDLE_WITH_XPU
    } else if (XPUStorageProperties::classof(sp.get())) {
      auto result = std::make_unique<XPUStorageProperties>();
      result->xpu_scale_value =
          static_cast<XPUStorageProperties*>(sp.get())->xpu_scale_value;
      return result;
#endif
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

}  // namespace phi
