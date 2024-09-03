/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <mutex>  // NOLINT

#include "afs_api/include/afs_api_so.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag afsapi_dso_flag;
extern void* afsapi_dso_handle;

#define DYNAMIC_LOAD_AFSAPI_WRAP(__name)                             \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using afsapiFunc = decltype(&::__name);                        \
      std::call_once(afsapi_dso_flag, []() {                         \
        afsapi_dso_handle = phi::dynload::GetAfsApiDsoHandle();      \
      });                                                            \
      static void* p_##__name = dlsym(afsapi_dso_handle, #__name);   \
      return reinterpret_cast<afsapiFunc>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_AFSAPI_WRAP(__name) \
  DYNAMIC_LOAD_AFSAPI_WRAP(__name)

#define AFSAPI_ROUTINE_EACH(__macro) \
  __macro(afs_init);                 \
  __macro(afs_open_writer);          \
  __macro(afs_open_reader);          \
  __macro(afs_close_reader);         \
  __macro(afs_close_writer);         \
  __macro(afs_writer_write);         \
  __macro(afs_writer_write_v2);      \
  __macro(afs_reader_read);          \
  __macro(afs_touchz);               \
  __macro(afs_mv);                   \
  __macro(afs_remove);               \
  __macro(afs_mkdir);                \
  __macro(afs_download_file);        \
  __macro(afs_upload_file);          \
  __macro(afs_list);                 \
  __macro(afs_cat);                  \
  __macro(afs_exist);                \
  __macro(afs_free);                 \
  __macro(createAfsAPIWrapper);      \
  __macro(destroyAfsAPIWrapper);

AFSAPI_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_AFSAPI_WRAP);

#undef DYNAMIC_LOAD_AFSAPI_WRAP

}  // namespace dynload
}  // namespace phi
