<<<<<<< HEAD
#pragma once

#include <string>
#include <stdexcept>

#if !defined(_WIN32)
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#else
#define NOMINMAX // windows min(), max() macro will mess std::min,max
#include <Shlwapi.h>
#include <Windows.h>
namespace {

static void* dlsym(void *handle, const char* symbol_name) {
	FARPROC found_symbol;
    found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

    if (found_symbol == NULL) {
    	throw std::runtime_error(std::string(symbol_name) + " not found.");
    }
    return (void*)found_symbol;
}
} // namespace anoymous

#endif
=======
// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdexcept>
#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#else
#include <Shlwapi.h>
#include <Windows.h>

static void* dlsym(void* handle, const char* symbol_name) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

  if (found_symbol == NULL) {
    throw std::runtime_error(std::string(symbol_name) + " not found.");
  }
  return reinterpret_cast<void*>(found_symbol);
}

#endif
>>>>>>> origin/develop
