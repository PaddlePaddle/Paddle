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

#include <memory>
#include <string>

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "glog/logging.h"

#if !defined(_WIN32)
#define UNUSED __attribute__((unused))
#include <dlfcn.h>     //  dladdr
#include <execinfo.h>  // backtrace
#include <sys/stat.h>
#include <algorithm>  // std::accumulate
#else
#include <io.h>  // _popen, _pclose
#include <windows.h>
#if defined(_WIN32)
#include <numeric>  // std::accumulate in msvc
#endif
// windows version of __attribute__((unused))
#define UNUSED __pragma(warning(suppress : 4100))

#ifndef S_ISDIR  // windows port for sys/stat.h
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif  // S_ISDIR

static void *dlsym(void *handle, const char *symbol_name) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

  if (found_symbol == NULL) {
    throw std::runtime_error(std::string(symbol_name) + " not found.");
  }
  return reinterpret_cast<void *>(found_symbol);
}

static void *dlopen(const char *filename, int flag) {
  std::string file_name(filename);
  file_name.replace(0, file_name.size() - 1, '/', '\\');
  HMODULE hModule = LoadLibrary(file_name.c_str());
  if (!hModule) {
    throw std::runtime_error(file_name + " not found.");
  }
  return reinterpret_cast<void *>(hModule);
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

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
#if !defined(_WIN32)
constexpr char kSEP = '/';
#else
constexpr char kSEP = '\\';
#endif  // _WIN32

static bool FileExists(const std::string &filepath) {
#if !defined(_WIN32)
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
#else
  struct _stat buffer;
  return (_stat(filepath.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static void MkDir(const char *path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
#if !defined(_WIN32)
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
#else
  CreateDirectory(path, NULL);
  auto errorno = GetLastError();
  if (errorno != ERROR_ALREADY_EXISTS) {
    throw std::runtime_error(path_error);
  }
#endif  // !_WIN32
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}
