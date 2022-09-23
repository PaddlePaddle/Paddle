// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(_WIN32)
#ifndef PADDLE_API
#ifdef PADDLE_DLL_EXPORT
#define PADDLE_API __declspec(dllexport)
#else
#define PADDLE_API __declspec(dllimport)
#endif  // PADDLE_DLL_EXPORT
#endif  // PADDLE_API
#else
#define PADDLE_API
#endif  // _WIN32
