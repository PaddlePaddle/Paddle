/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {

#define ONE_CASE(key) \
  case key:           \
    return #key

const char* to_string(KernelType kt) {
  switch (kt) {
    ONE_CASE(vmul);
    ONE_CASE(vadd);
    ONE_CASE(vaddrelu);
    ONE_CASE(vsub);
    ONE_CASE(vscal);
    ONE_CASE(vaddbias);
    ONE_CASE(vrelu);
    ONE_CASE(videntity);
    ONE_CASE(vexp);
    ONE_CASE(vsigmoid);
    ONE_CASE(vtanh);
    default:
      PADDLE_THROW("Not support type: %d", kt);
      return "NOT JITKernel";
  }
  return nullptr;
}
#undef ONE_CASE

}  // namespace jit
}  // namespace operators
}  // namespace paddle
