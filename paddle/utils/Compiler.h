/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
/**
 * This header defines some useful attribute by each compiler. It is the
 * abstract layer of compilers.
 */
#ifdef __GNUC__
#define GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
#define GCC_VERSION
#endif

/**
 * __must_check macro. It make the function's return value must be used,
 * otherwise it will raise a compile warning. And also Paddle treat all compile
 * warnings as errors.
 */
#if GCC_VERSION >= 30400
#define __must_check __attribute__((warn_unused_result))
#else
#define __must_check
#endif
