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

#include "paddle/fluid/operators/math/vol2col.h"

#include <gtest/gtest.h>
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

template <typename DeviceContext, typename Place>
void testVol2col() {
  paddle::framework::Tensor input;
  paddle::framework::Tensor input_tmp;
  paddle::framework::Tensor output;
  paddle::framework::Tensor output_tmp;

  auto* place = new Place();
  DeviceContext* context = new DeviceContext(*place);
  /**
   * input = [[0, 1, 2,
   *          3, 4, 5]
   *          [6, 7, 8,
   *          9, 10, 11]]
   *
   * output = [0, 1
   *           1, 2
   *           3, 4
   *           4, 5
   *           6, 7
   *           7, 8
   *           9, 10
   *           10, 11]
   *
   * col2vol = [[0, 2, 2,
   *             3, 8, 5]
   *            [6, 14, 8,
   *             9, 20, 11]]
   *
   */
  int input_depth = 2;
  int input_height = 2;
  int input_width = 3;
  int filter_size = 2;
  std::vector<int> strides({1, 1, 1});
  std::vector<int> paddings({0, 0, 0});
  std::vector<int> dilations({1, 1, 1});
  int output_depth =
      (input_depth - filter_size + 2 * paddings[0]) / strides[0] + 1;
  int output_height =
      (input_height - filter_size + 2 * paddings[1]) / strides[1] + 1;
  int output_width =
      (input_width - filter_size + 2 * paddings[2]) / strides[2] + 1;

  // Vol2Col test
  float* input_ptr =
      input_tmp.mutable_data<float>({1, input_depth, input_height, input_width},
                                    paddle::platform::CPUPlace());
  float arr[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input_ptr, arr, 12 * sizeof(float));

  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }
  output.mutable_data<float>({1, filter_size, filter_size, filter_size,
                              output_depth, output_height, output_width},
                             *place);

  paddle::operators::math::Vol2ColFunctor<DeviceContext, float> vol2col;
  vol2col(*context, input, dilations, strides, paddings, &output);

  float vol_2_col[] = {0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11};
  float* out_cfo_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    out_cfo_ptr = output.data<float>();
  } else {
    paddle::framework::TensorCopySync(output, paddle::platform::CPUPlace(),
                                      &output_tmp);
    out_cfo_ptr = output_tmp.data<float>();
  }

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out_cfo_ptr[i], vol_2_col[i]);
  }

  // Col2Vol test
  float col_2_vol[] = {0, 2, 2, 3, 8, 5, 6, 14, 8, 9, 20, 11};
  memset(input_ptr, 0, 12 * sizeof(float));
  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }

  paddle::operators::math::Col2VolFunctor<DeviceContext, float> col2vol;
  col2vol(*context, output, dilations, strides, paddings, &input);

  float* in_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    in_ptr = input.data<float>();
  } else {
    paddle::framework::TensorCopySync(input, paddle::platform::CPUPlace(),
                                      &input_tmp);
    in_ptr = input_tmp.data<float>();
  }

  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(in_ptr[i], col_2_vol[i]);
  }

  delete place;
  delete context;
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
void testVol2col<paddle::platform::CUDADeviceContext,
                 paddle::platform::CUDAPlace>() {
  paddle::framework::Tensor input;
  paddle::framework::Tensor input_tmp;
  paddle::framework::Tensor output;
  paddle::framework::Tensor output_tmp;

  auto* place = new paddle::platform::CUDAPlace();
  auto* context = new paddle::platform::CUDADeviceContext(*place);
  context->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                            .GetAllocator(*place, context->stream())
                            .get());
  context->PartialInitWithAllocator();

  /**
   * input = [[0, 1, 2,
   *          3, 4, 5]
   *          [6, 7, 8,
   *          9, 10, 11]]
   *
   * output = [0, 1
   *           1, 2
   *           3, 4
   *           4, 5
   *           6, 7
   *           7, 8
   *           9, 10
   *           10, 11]
   *
   * col2vol = [[0, 2, 2,
   *             3, 8, 5]
   *            [6, 14, 8,
   *             9, 20, 11]]
   *
   */
  int input_depth = 2;
  int input_height = 2;
  int input_width = 3;
  int filter_size = 2;
  std::vector<int> strides({1, 1, 1});
  std::vector<int> paddings({0, 0, 0});
  std::vector<int> dilations({1, 1, 1});
  int output_depth =
      (input_depth - filter_size + 2 * paddings[0]) / strides[0] + 1;
  int output_height =
      (input_height - filter_size + 2 * paddings[1]) / strides[1] + 1;
  int output_width =
      (input_width - filter_size + 2 * paddings[2]) / strides[2] + 1;

  // Vol2Col test
  float* input_ptr =
      input_tmp.mutable_data<float>({1, input_depth, input_height, input_width},
                                    paddle::platform::CPUPlace());
  float arr[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input_ptr, arr, 12 * sizeof(float));

  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }
  output.mutable_data<float>({1, filter_size, filter_size, filter_size,
                              output_depth, output_height, output_width},
                             *place);

  paddle::operators::math::Vol2ColFunctor<paddle::platform::CUDADeviceContext,
                                          float>
      vol2col;
  vol2col(*context, input, dilations, strides, paddings, &output);

  float vol_2_col[] = {0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11};
  float* out_cfo_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    out_cfo_ptr = output.data<float>();
  } else {
    paddle::framework::TensorCopySync(output, paddle::platform::CPUPlace(),
                                      &output_tmp);
    out_cfo_ptr = output_tmp.data<float>();
  }

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out_cfo_ptr[i], vol_2_col[i]);
  }

  // Col2Vol test
  float col_2_vol[] = {0, 2, 2, 3, 8, 5, 6, 14, 8, 9, 20, 11};
  memset(input_ptr, 0, 12 * sizeof(float));
  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }

  paddle::operators::math::Col2VolFunctor<paddle::platform::CUDADeviceContext,
                                          float>
      col2vol;
  col2vol(*context, output, dilations, strides, paddings, &input);

  float* in_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    in_ptr = input.data<float>();
  } else {
    paddle::framework::TensorCopySync(input, paddle::platform::CPUPlace(),
                                      &input_tmp);
    in_ptr = input_tmp.data<float>();
  }

  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(in_ptr[i], col_2_vol[i]);
  }

  delete place;
  delete context;
}
#endif

TEST(math, vol2col) {
  testVol2col<paddle::platform::CPUDeviceContext, paddle::platform::CPUPlace>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  testVol2col<paddle::platform::CUDADeviceContext,
              paddle::platform::CUDAPlace>();
#endif  // PADDLE_WITH_CUDA
}
