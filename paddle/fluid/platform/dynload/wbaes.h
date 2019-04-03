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

#ifdef PADDLE_WITH_WBAES

#include <WBAESLib.h>
#include <string.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag wbaes_dso_flag;
extern void *wbaes_dso_handle;
extern void *wbaes_func[9];  // 9 is number given by library provider

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load wbaes routine
 * via operator overloading.
 */

#define DYNAMIC_LOAD_WBAES_WRAP(__name)                                    \
  struct DynLoad__##__name {                                               \
    void *operator[](int i) {                                              \
      std::call_once(wbaes_dso_flag, []() {                                \
        wbaes_dso_handle = paddle::platform::dynload::GetWBAESDsoHandle(); \
        static void *p_##__name = dlsym(wbaes_dso_handle, #__name);        \
        memcpy(wbaes_func, p_##__name, sizeof(wbaes_func));                \
      });                                                                  \
      return wbaes_func[i];                                                \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WBAES_WRAP(__name) DYNAMIC_LOAD_WBAES_WRAP(__name)

#define WBAES_ROUTINE_EACH(__macro) __macro(GSECF);

WBAES_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WBAES_WRAP);

typedef int (*FN0)(const char *, const char *);
#define WBAESInit(encryptTable, decryptTable) \
  ((FN0)platform::dynload::GSECF[0])(encryptTable, decryptTable)

typedef int (*FN2)(const char *, char *, const long);  // NOLINT
#define WBAESEncrypt(input, output, length) \
  ((FN2)platform::dynload::GSECF[2])(input, output, length)

typedef int (*FN3)(const char *, char *, const long);  // NOLINT
#define WBAESDecrypt(input, output, length) \
  ((FN3)platform::dynload::GSECF[3])(input, output, length)

typedef int (*FN4)(const char *, const char *, const int);
#define WBAESEncryptFile(inPath, outPath, blockSize) \
  ((FN4)platform::dynload::GSECF[4])(inPath, outPath, blockSize)

typedef int (*FN5)(const char *, const char *, const int);
#define WBAESDecryptFile(inPath, outPath, blockSize) \
  ((FN5)platform::dynload::GSECF[5])(inPath, outPath, blockSize)

typedef int (*FN7)(const char *key, const char *encryptTablePath,
                   const char *decryptTablePath);
#define WBAESCreateKeyInFile(key, encryptTablePath, decryptTablePath) \
  ((FN7)platform::dynload::GSECF[7])(key, encryptTablePath, decryptTablePath)

#undef DYNAMIC_LOAD_WBAES_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif
