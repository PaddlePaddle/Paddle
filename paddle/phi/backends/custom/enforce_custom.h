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
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include <string>

#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
template <typename T>
struct CustomDeviceStatusType {};

#define DEFINE_CUSTOM_DEVICE_STATUS_TYPE(type, success_value) \
  template <>                                                 \
  struct CustomDeviceStatusType<type> {                       \
    using Type = type;                                        \
    static constexpr Type kSuccess = success_value;           \
  }

DEFINE_CUSTOM_DEVICE_STATUS_TYPE(C_Status, C_SUCCESS);

inline std::string build_custom_device_error_msg(C_Status stat) {
  std::ostringstream sout;
  sout << " CustomDevice error, the error code is : " << stat << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_CUSTOM_DEVICE_SUCCESS(COND)                   \
  do {                                                               \
    auto __cond__ = (COND);                                          \
    using __CUSTOM_DEVICE_STATUS_TYPE__ = decltype(__cond__);        \
    constexpr auto __success_type__ = ::phi::CustomDeviceStatusType< \
        __CUSTOM_DEVICE_STATUS_TYPE__>::kSuccess;                    \
    if (UNLIKELY(__cond__ != __success_type__)) {                    \
      auto __summary__ = ::common::errors::External(                 \
          ::phi::build_custom_device_error_msg(__cond__));           \
      __THROW_ERROR_INTERNAL__(__summary__);                         \
    }                                                                \
  } while (0)
}  // namespace phi
#endif  // PADDLE_WITH_CUSTOM_DEVICE
