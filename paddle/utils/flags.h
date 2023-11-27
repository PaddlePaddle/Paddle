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

#pragma once

#ifdef PADDLE_WITH_GFLAGS
#include "gflags/gflags.h"
#else
#include "paddle/utils/flags_native.h"
#endif

#ifdef PADDLE_WITH_GFLAGS
#define PD_DEFINE_bool(name, val, txt) DEFINE_bool(name, val, txt)
#define PD_DEFINE_int32(name, val, txt) DEFINE_int32(name, val, txt)
#define PD_DEFINE_uint32(name, val, txt) DEFINE_uint32(name, val, txt)
#define PD_DEFINE_int64(name, val, txt) DEFINE_int64(name, val, txt)
#define PD_DEFINE_uint64(name, val, txt) DEFINE_uint64(name, val, txt)
#define PD_DEFINE_double(name, val, txt) DEFINE_double(name, val, txt)
#define PD_DEFINE_string(name, val, txt) DEFINE_string(name, val, txt)

#define PD_DECLARE_bool(name) DECLARE_bool(name)
#define PD_DECLARE_int32(name) DECLARE_int32(name)
#define PD_DECLARE_uint32(name) DECLARE_uint32(name)
#define PD_DECLARE_int64(name) DECLARE_int64(name)
#define PD_DECLARE_uint64(name) DECLARE_uint64(name)
#define PD_DECLARE_double(name) DECLARE_double(name)
#define PD_DECLARE_string(name) DECLARE_string(name)
#endif

namespace paddle {
namespace flags {

#ifdef PADDLE_WITH_GFLAGS
inline void ParseCommandLineFlags(int* argc, char*** argv) {
  gflags::ParseCommandLineFlags(argc, argv, true);
}
#else
using paddle::flags::ParseCommandLineFlags;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline bool SetFlagValue(const char* name, const char* value) {
  std::string ret = gflags::SetCommandLineOption(name, value);
  return ret.empty() ? false : true;
}
#else
using paddle::flags::SetFlagValue;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline bool FindFlag(const char* name) {
  std::string value;
  return gflags::GetCommandLineOption(name, &value);
}
#else
using paddle::flags::FindFlag;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline void AllowUndefinedFlags() { gflags::AllowCommandLineReparsing(); }
#else
using paddle::flags::AllowUndefinedFlags;
#endif

#ifdef PADDLE_WITH_GFLAGS
using gflags::BoolFromEnv;
using gflags::DoubleFromEnv;
using gflags::Int32FromEnv;
using gflags::Int64FromEnv;
using gflags::StringFromEnv;
using gflags::Uint32FromEnv;
using gflags::Uint64FromEnv;
#else
#define DEFINE_FROM_ENV_FUNC(type, name)                     \
  inline type name##FromEnv(const std::string& env_var_name, \
                            type default_val) {              \
    return GetFromEnv(env_var_name, default_val);            \
  }

DEFINE_FROM_ENV_FUNC(bool, Bool);
DEFINE_FROM_ENV_FUNC(int32_t, Int32);
DEFINE_FROM_ENV_FUNC(uint32_t, Uint32);
DEFINE_FROM_ENV_FUNC(int64_t, Int64);
DEFINE_FROM_ENV_FUNC(uint64_t, Uint64);
DEFINE_FROM_ENV_FUNC(double, Double);
DEFINE_FROM_ENV_FUNC(std::string, String);

#undef DEFINE_FROM_ENV_FUNC
#endif

}  // namespace flags
}  // namespace paddle
