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

#include <cstdio>
#include <stdexcept>

#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#include <sys/stat.h>
#else
#include <io.h>  // _popen, _pclose
#include <windows.h>

#ifndef S_ISDIR  // windows port for sys/stat.h
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif

static void* dlsym(void* handle, const char* symbol_name) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

  if (found_symbol == NULL) {
    throw std::runtime_error(std::string(symbol_name) + " not found.");
  }
  return reinterpret_cast<void*>(found_symbol);
}

#endif  // !_WIN32

static void ExecShellCommand(const std::string &cmd, std::string *message) {
  char buffer[128];
#if !defined(_WIN32)
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
#else
  std::shared_ptr<FILE> pipe(_popen(cmd.c_str(), "r"), _pclose);
#endif  // _WIN32
  if (!pipe) {
    LOG(ERROR) << "error running command: " << cmd;
    return;
  }
  while (!feof(pipe.get())) {
    if (fgets(buffer, 128, pipe.get()) != nullptr) {
      *message += buffer;
    }
  }
}

static bool PathExists(const std::string &path) {
#if !defined(_WIN32)
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
#else
  struct _stat statbuf;
  if (_stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
#endif  // !_WIN32
  return false;
}

static FILE
