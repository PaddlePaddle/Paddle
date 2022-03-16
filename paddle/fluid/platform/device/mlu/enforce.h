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

#include "paddle/fluid/platform/enforce.h"
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#endif  // PADDLE_WITH_MLU

#ifdef PADDLE_WITH_MLU
DECLARE_int64(gpu_allocator_retry_time);
#endif

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_MLU
namespace details {
template <typename T>
struct MLUStatusType {};

#define DEFINE_MLU_STATUS_TYPE(type, success_value, proto_type) \
  template <>                                                   \
  struct MLUStatusType<type> {                                  \
    using Type = type;                                          \
    static constexpr Type kSuccess = success_value;             \
    static constexpr const char* kTypeString = #proto_type;     \
  }

DEFINE_MLU_STATUS_TYPE(cnrtStatus, cnrtSuccess, CNRT);
DEFINE_MLU_STATUS_TYPE(cnnlStatus, CNNL_STATUS_SUCCESS, CNNL);
DEFINE_MLU_STATUS_TYPE(cnStatus, CN_SUCCESS, CN);
#ifdef PADDLE_WITH_CNCL
DEFINE_MLU_STATUS_TYPE(cnclStatus, CNCL_RET_SUCCESS, CNCL);
#endif

}  // namespace details

/*************** CNRT ERROR ***************/
inline bool is_error(cnrtStatus e) { return e != cnrtSuccess; }

inline std::string build_mlu_error_msg(cnrtStatus e) {
  std::ostringstream sout;
  sout << "MLU CNRT error(" << e << "), " << cnrtGetErrorName(e) << ": "
       << cnrtGetErrorStr(e);
  return sout.str();
}

/*************** CNNL ERROR ***************/
inline bool is_error(cnnlStatus stat) { return stat != CNNL_STATUS_SUCCESS; }

inline std::string build_mlu_error_msg(cnnlStatus stat) {
  std::ostringstream sout;
  sout << "MLU CNNL error(" << stat << "), " << cnnlGetErrorString(stat)
       << ". ";
  return sout.str();
}

/*************** CN API ERROR ***************/
inline bool is_error(cnStatus stat) { return stat != CN_SUCCESS; }

inline std::string build_mlu_error_msg(cnStatus stat) {
  const char* error_name;
  const char* error_string;
  cnGetErrorName(stat, &error_name);
  cnGetErrorString(stat, &error_string);

  std::ostringstream sout;
  sout << "MLU CN error(" << static_cast<int>(stat) << "), " << error_name
       << " : " << error_string << ". ";
  return sout.str();
}

/*************** CNCL ERROR ***************/
#ifdef PADDLE_WITH_CNCL
inline bool is_error(cnclStatus e) { return e != CNCL_RET_SUCCESS; }

inline std::string build_mlu_error_msg(cnclStatus e) {
  std::ostringstream sout;
  sout << "MLU CNCL error(" << e << "), " << cnclGetErrorStr(e) << ". ";
  return sout.str();
}
#endif

#define PADDLE_ENFORCE_MLU_SUCCESS(COND)                       \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __MLU_STATUS_TYPE__ = decltype(__cond__);            \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::MLUStatusType<            \
            __MLU_STATUS_TYPE__>::kSuccess;                    \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      auto __summary__ = ::paddle::platform::errors::External( \
          ::paddle::platform::build_mlu_error_msg(__cond__));  \
      __THROW_ERROR_INTERNAL__(__summary__);                   \
    }                                                          \
  } while (0)

#define PADDLE_ENFORCE_MLU_LAUNCH_SUCCESS(OP)                                  \
  do {                                                                         \
    auto res = cnrtGetLastError();                                             \
    if (UNLIKELY(res != cnrtSuccess)) {                                        \
      auto msg = ::paddle::platform::build_mlu_error_msg(res);                 \
      PADDLE_THROW(platform::errors::Fatal("CNRT error after kernel (%s): %s", \
                                           OP, msg));                          \
    }                                                                          \
  } while (0)

inline void retry_sleep(unsigned milliseconds) {
  if (milliseconds < 1000) {
    // usleep argument must be less than 1,000,000. Reference:
    // https://pubs.opengroup.org/onlinepubs/7908799/xsh/usleep.html
    usleep(milliseconds * 1000);
  } else {
    // clip to sleep in seconds because we can not and don't have to
    // sleep for exact milliseconds
    sleep(milliseconds / 1000);
  }
}

#define PADDLE_RETRY_MLU_SUCCESS(COND)                                  \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __MLU_STATUS_TYPE__ = decltype(__cond__);                     \
    constexpr auto __success_type__ =                                   \
        ::paddle::platform::details::MLUStatusType<                     \
            __MLU_STATUS_TYPE__>::kSuccess;                             \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      retry_sleep(FLAGS_gpu_allocator_retry_time);                      \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = ::paddle::platform::errors::External(          \
          ::paddle::platform::build_mlu_error_msg(__cond__));           \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_MLU_STATUS_TYPE
#endif  // PADDLE_WITH_MLU

}  // namespace platform
}  // namespace paddle
