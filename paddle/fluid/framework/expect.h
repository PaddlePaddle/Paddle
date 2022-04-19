// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#ifndef likely
#ifdef _LINUX
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define likely(expr) (expr)
#endif
#endif

#ifndef unlikely
#ifdef _LINUX
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#else
#define unlikely(expr) (expr)
#endif
#endif
