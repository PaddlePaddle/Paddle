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
#include "paddle/framework/selected_rows.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

// SelectedRows + SelectedRows will simplely concat value and rows.
// The real computation happens in dealing with LoDTensor.
template <typename Place, typename T>
struct SelectedRowsAdd {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output);
};

template <typename Place, typename T>
struct SelectedRowsAddTensor {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output);
};

// input2 = input1 + input2
template <typename Place, typename T>
struct SelectedRowsAddTo {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const int64_t input2_offset, framework::SelectedRows* input2);
};

// input2 = input1 + input2
template <typename Place, typename T>
struct SelectedRowsAddToTensor {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
