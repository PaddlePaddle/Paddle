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

#pragma once

#ifdef PADDLE_WITH_ASCEND_CL
#include <string>

#include "paddle/fluid/platform/enforce.h"

#include "acl/acl.h"
#include "hccl/hccl_types.h"

namespace paddle {
namespace platform {

namespace details {
template <typename T>
struct NPUStatusType {};

#define DEFINE_NPU_STATUS_TYPE(type, success_value) \
  template <>                                       \
  struct NPUStatusType<type> {                      \
    using Type = type;                              \
    static constexpr Type kSuccess = success_value; \
  }

DEFINE_NPU_STATUS_TYPE(aclError, ACL_ERROR_NONE);
DEFINE_NPU_STATUS_TYPE(HcclResult, HCCL_SUCCESS);
}  // namespace details

inline std::string build_npu_error_msg(aclError stat) {
  std::ostringstream sout;
  sout << " ACL error, the error code is : " << stat << ". ";
  return sout.str();
}

inline std::string build_npu_error_msg(HcclResult stat) {
  std::ostringstream sout;
  sout << " HCCL error, the error code is : " << stat << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_NPU_SUCCESS(COND)                       \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __NPU_STATUS_TYPE__ = decltype(__cond__);            \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::NPUStatusType<            \
            __NPU_STATUS_TYPE__>::kSuccess;                    \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      auto __summary__ = ::paddle::platform::errors::External( \
          ::paddle::platform::build_npu_error_msg(__cond__));  \
      __THROW_ERROR_INTERNAL__(__summary__);                   \
    }                                                          \
  } while (0)

}  // namespace platform
}  // namespace paddle
#endif  // PADDLE_WITH_ASCEND_CL
