// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <paddle/phi/common/port.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include "glog/logging.h"

#if !defined(_WIN32)
#include <dlfcn.h>  // dladdr
#include <sys/stat.h>
#include <sys/time.h>

#else
#include <numeric>  // std::accumulate in msvc

void *dlsym(void *handle, const char *symbol_name) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

  if (found_symbol == NULL) {
    LOG(ERROR) << "Load symbol " << symbol_name << " failed.";
    throw std::runtime_error(std::string(symbol_name) + " not found.");
  }
  return reinterpret_cast<void *>(found_symbol);
}

void *dlopen(const char *filename, int flag) {
  std::string file_name(filename);
  HMODULE hModule = nullptr;
#ifdef WITH_PIP_CUDA_LIBRARIES
  hModule =
      LoadLibraryEx(file_name.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#endif
  if (!hModule) {
    hModule = LoadLibrary(file_name.c_str());
  }
  if (!hModule) {
    if (flag) {
      throw std::runtime_error(file_name + " not found.");
    } else {
      return nullptr;
    }
  }
  return reinterpret_cast<void *>(hModule);
}

int gettimeofday(struct timeval *tp, void *tzp) {
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;

  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;

  return (0);
}
#endif  // !_WIN32

void ExecShellCommand(const std::string &cmd, std::string *message) {
  std::array<char, 128> buffer = {};
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
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr) {
      *message += buffer.data();
    }
  }
}

bool PathExists(const std::string &path) {
#if !defined(_WIN32)
  struct stat statbuf = {};
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

#if !defined(_WIN32)
constexpr char kSEP = '/';
#else
constexpr char kSEP = '\\';
#endif  // _WIN32

bool FileExists(const std::string &filepath) {
#if !defined(_WIN32)
  struct stat buffer = {};
  return (stat(filepath.c_str(), &buffer) == 0);
#else
  struct _stat buffer;
  return (_stat(filepath.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

void MkDir(const char *path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
#if !defined(_WIN32)
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
#else
  BOOL return_value = CreateDirectory(path, NULL);
  if (!return_value) {
    auto errorno = GetLastError();
    if (errorno != ERROR_ALREADY_EXISTS) {
      throw std::runtime_error(path_error);
    }
  }
#endif  // !_WIN32
}

void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;

  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}
