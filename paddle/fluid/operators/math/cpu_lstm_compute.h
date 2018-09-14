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
#include <string>

namespace paddle {
namespace operators {
namespace math {

// TODO(TJ): ugly workaround, clean me
template <typename T>
void lstm_compute_ctht(T* gates, const T* ct_1, T* ct, T* ht);

}  // namespace math
}  // namespace operators
}  // namespace paddle
