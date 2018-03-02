/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#ifndef __ANDROID__
#include <execinfo.h>
#else
#include <unwind.h>
#include <iomanip>

// See the introduction on https://linux.die.net/man/3/backtrace_symbols
// The implementation for Android is modified from the public implementation on
// https://stackoverflow.com/questions/8115192/android-ndk-getting-the-backtrace

namespace paddle {
namespace platform {

struct BacktraceState {
  void** current;
  void** end;
};

static _Unwind_Reason_Code UnwindCallback(struct _Unwind_Context* context,
                                          void* arg) {
  BacktraceState* state = static_cast<BacktraceState*>(arg);
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    if (state->current == state->end) {
      return _URC_END_OF_STACK;
    } else {
      *state->current++ = reinterpret_cast<void*>(pc);
    }
  }
  return _URC_NO_REASON;
}

static int backtrace(void** buffer, int size) {
  BacktraceState state = {buffer, buffer + size};
  _Unwind_Backtrace(UnwindCallback, &state);

  return state.current - buffer;
}

static char** backtrace_symbols(void* const* buffer, int size) {
  if ((buffer == nullptr) || (*buffer == nullptr) || size <= 0) {
    return nullptr;
  }
  char** symbols = (char**)malloc(size * sizeof(char*));
  if (symbols == nullptr) {
    return nullptr;
  }
  for (int i = 0; i < size; ++i) {
    symbols[i] = reinterpret_cast<char*>(buffer[i]);
  }
  return symbols;
}

}  // namespace platform
}  // namespace paddle

#endif
