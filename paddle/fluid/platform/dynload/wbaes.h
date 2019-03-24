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

#include <WBAESLib.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag wbaes_dso_flag;
extern void *wbaes_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load wbaes routine
 * via operator overloading.
 */

#define DYNAMIC_LOAD_WBAES_WRAP(__name)                                    \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {       \
      using wbaesFunc = decltype(&::__name);                               \
      std::call_once(wbaes_dso_flag, []() {                                \
        wbaes_dso_handle = paddle::platform::dynload::GetWBAESDsoHandle(); \
      });                                                                  \
      static void *p_##__name = dlsym(wbaes_dso_handle, #__name);          \
      return reinterpret_cast<wbaesFunc>(p_##__name)(args...);             \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WBAES_WRAP(__name) DYNAMIC_LOAD_WBAES_WRAP(__name)

#define WBAES_ROUTINE_EACH(__macro) __macro(GSECF);

WBAES_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WBAES_WRAP);

typedef int (*FN0)(const char *, const char *);
#define wbaes_init(encryptTable, decryptTable) \
  ((FN0)GSECF[0])(encryptTable, decryptTable)

typedef int (*FN1)(const char *, const char *);
#define wbaes_init_memory(encryptTable, decryptTable) \
  ((FN1)GSECF[1])(encryptTable, decryptTable)

typedef int (*FN2)(const char *, char *, const long);  // NOLINT
#define wbaes_encrypt(input, output, length) \
  ((FN2)GSECF[2])(input, output, length)

typedef int (*FN3)(const char *, char *, const long);  // NOLINT
#define wbaes_decrypt(input, output, length) \
  ((FN3)GSECF[3])(input, output, length)

typedef int (*FN4)(const char *, const char *, const int);
#define wbaes_encrypt_file(inPath, outPath, blockSize) \
  ((FN4)GSECF[4])(inPath, outPath, blockSize)

typedef int (*FN5)(const char *, const char *, const int);
#define wbaes_decrypt_file(inPath, outPath, blockSize) \
  ((FN5)GSECF[5])(inPath, outPath, blockSize)

typedef int (*FN6)(const char *key, char **encryptTable,
                   int *encryptTableLength, char **decryptTable,
                   int *decryptTableLength);
#define wbaes_create_key_in_memory(key, encryptTable, encryptTableLength, \
                                   decryptTable, decryptTableLength)      \
  ((FN6)GSECF[6])(key, encryptTable, encryptTableLength, decryptTable,    \
                  decryptTableLength)

typedef int (*FN7)(const char *key, const char *encryptTablePath,
                   const char *decryptTablePath);
#define wbaes_create_key_in_file(key, encryptTablePath, decryptTablePath) \
  ((FN7)GSECF[7])(key, encryptTablePath, decryptTablePath)

#undef DYNAMIC_LOAD_WBAES_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
