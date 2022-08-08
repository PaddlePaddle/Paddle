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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"

#define MAX_UINT32_VALUE 0xFFFFFFFF
#define INVALID_BFID -1

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_XPU_KP)

class CacheManagerKernel {
public:
  CacheManagerKernel() { }
  ~CacheManagerKernel() { }

#if defined(PADDLE_WITH_XPU_CACHE_BFID)
  void convert_fid2bfid(uint32_t * fid_seq_ptr,
                        uint32_t fid_seq_size,
                        uint32_t * bucket_sizes,
                        uint32_t bucket_num,
                        uint32_t * key_in_ptr,
                        uint32_t key_in_size,
                        int * key_out_ptr,
                        ppStream & stream);

#endif
};

#endif // end PADDLE_WITH_XPU_KP
}  // end namespace framework
}  // end namespace paddle
#endif // end PADDLE_WITH_HETERPS
