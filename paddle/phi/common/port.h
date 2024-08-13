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

#include <string>
#include "paddle/utils/test_macros.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h

#if !defined(_WIN32)
#include <dlfcn.h>  // dladdr
#include <sys/time.h>

#else
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
// solve static linking error in windows
// https://github.com/google/glog/issues/301
#define GOOGLE_GLOG_DLL_DECL
#include <io.h>  // _popen, _pclose
#include <stdio.h>
#include <windows.h>
#include <winsock.h>

#ifndef S_ISDIR  // windows port for sys/stat.h
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)
#endif  // S_ISDIR

TEST_API void *dlsym(void *handle, const char *symbol_name);

void *dlopen(const char *filename, int flag);

int gettimeofday(struct timeval *tp, void *tzp);
#endif  // !_WIN32

void ExecShellCommand(const std::string &cmd, std::string *message);

bool PathExists(const std::string &path);

// TODO(yuyang18): If the functions below are needed by other files, move them
// to paddle::filesystem namespace.
bool FileExists(const std::string &filepath);

std::string DirName(const std::string &filepath);

void MkDir(const char *path);

void MkDirRecursively(const char *fullpath);
