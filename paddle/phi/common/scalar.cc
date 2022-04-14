/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/scalar.h"

#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

// NOTE(xiongkun): why we put definition here?
// test_custom_op can't include enforce.h, because enforce.h includes gflags.
// so we decouple the include dependence of enforce.h by link.
void ThrowTensorConvertError(int num) {
  PADDLE_ENFORCE_EQ(num,
                    1,
                    phi::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, but "
                        "now Tensor has `%d` elements",
                        num));
}

}  // namespace experimental
}  // namespace paddle
