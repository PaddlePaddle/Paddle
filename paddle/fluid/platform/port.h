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
#define UNUSED __attribute__((unused))

#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#include <sys/stat.h>

#else
#include <io.h>  // _popen, _pclose
#include <windows.h>

// windows version of __attribute__((unused))
#define UNUSED __pragma(warning(suppress : 4100))

#ifndef S_ISDIR  // windows port for sys/stat.h
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif

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
  std::replace(file_name.begin(), file_name.end(), '/', '\\');
  HMODULE hModule = LoadLibrary(file_name);
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
#if !defined(_WIN32)
  if (mkdir(path, 0755)) {
    PADDLE_ENFORCE_EQ(errno, EEXIST, "%s mkdir failed!", path);
  }
#else
  CreateDirectory(path, NULL);
  auto errorno = GetLastError();
  PADDLE_ENFORCE_EQ(errorno, ERROR_ALREADY_EXISTS, "%s mkdir failed!", path);
#endif  // !_WIN32
}

static void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}
