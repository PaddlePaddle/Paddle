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
#include "paddle/utils/flags.h"
#else
#include "paddle/utils/flags_native.h"
#endif

#ifndef PADDLE_WITH_GFLAGS

#define DEFINE_bool(name, val, txt) PD_DEFINE_bool(name, val, txt)
#define DEFINE_int32(name, val, txt) PD_DEFINE_int32(name, val, txt)
#define DEFINE_uint32(name, val, txt) PD_DEFINE_uint32(name, val, txt)
#define DEFINE_int64(name, val, txt) PD_DEFINE_int64(name, val, txt)
#define DEFINE_uint64(name, val, txt) PD_DEFINE_uint64(name, val, txt)
#define DEFINE_double(name, val, txt) PD_DEFINE_double(name, val, txt)
#define DEFINE_string(name, val, txt) PD_DEFINE_string(name, val, txt)

#define DECLARE_bool(name) PD_DECLARE_bool(name)
#define DECLARE_int32(name) PD_DECLARE_int32(name)
#define DECLARE_uint32(name) PD_DECLARE_uint32(name)
#define DECLARE_int64(name) PD_DECLARE_int64(name)
#define DECLARE_uint64(name) PD_DECLARE_uint64(name)
#define DECLARE_double(name) PD_DECLARE_double(name)
#define DECLARE_string(name) PD_DECLARE_string(name)

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

}  // namespace flags
}  // namespace paddle
