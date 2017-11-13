/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <mutex>

namespace paddle {
namespace platform {

/*
 The current implementation of std::call_once has a bug described in
 https://stackoverflow.com/questions/41717579/stdcall-once-hangs-on-second-call-after-callable-threw-on-first-call.
 This is likely caused by a deeper bug of pthread_once, which is discussed in
 https://patchwork.ozlabs.org/patch/482350/

 This wrap is a hack to avoid this bug.
*/
template <typename Callable, typename... Args>
inline void call_once(std::once_flag& flag, Callable&& f, Args&&... args) {
  bool good = false;
  std::exception ex;
  std::call_once(flag,
                 [&](Args&&... args) {
                   try {
                     f(args...);
                     good = true;
                   } catch (const std::exception& e) {
                     ex = e;
                   } catch (...) {
                     ex = std::runtime_error("excption caught in call_once");
                   }
                 },
                 args...);
  if (!good) {
    throw std::exception(ex);
  }
}

}  // namespace platform
}  // namespace paddle
