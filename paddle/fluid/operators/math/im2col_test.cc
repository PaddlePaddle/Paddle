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

#include "paddle/fluid/operators/math/im2col.h"
#include <gtest/gtest.h>
#include "paddle/fluid/operators/math/im2col_cfo_cpu.h"

template <typename DeviceContext, typename Place>
void testIm2col() {
  paddle::framework::Tensor input_tmp;
  paddle::framework::Tensor input;
  paddle::framework::Tensor output_cfo;
  paddle::framework::Tensor output_ocf;
  paddle::framework::Tensor output_tmp;

  /**
   * input = [0, 1, 2,
   *          3, 4, 5]
   *
   * output_cfo = [0, 1
   *               1, 2
   *               3, 4
   *               4, 5]
   *
   * output_ocf = [0, 1, 3, 4
   *               1, 2, 4, 5]
   *
   * col2im_cfo = [0, 2, 2
   *               3, 4, 5]
   *
   * col2im_ocf = [0, 2, 2
   *               3, 4, 5]
   */
  int input_height = 2;
  int input_width = 3;
  int filter_size = 2;
  std::vector<int> stride({1, 1});  // stride_y, stride_x
  std::vector<int> padding(
      {0, 0, 0, 0});                  // up_pad, left_pad, down_pad, right_pad
  std::vector<int> dilation({1, 1});  // dilation_y, dilation_x
  int output_height =
      (input_height - filter_size + padding[0] + padding[1]) / stride[0] + 1;
  int output_width =
      (input_width - filter_size + padding[2] + padding[3]) / stride[1] + 1;
  float* input_ptr = input_tmp.mutable_data<float>(
      {1, input_height, input_width}, paddle::platform::CPUPlace());
  float arr[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input_ptr, arr, 6 * sizeof(float));

  auto* place = new Place();
  DeviceContext* context = new DeviceContext(*place);
  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }
  output_cfo.mutable_data<float>(
      {1, filter_size, filter_size, output_height, output_width}, *place);
  output_ocf.mutable_data<float>(
      {output_height, output_width, 1, filter_size, filter_size}, *place);

  // Im2Col
  paddle::operators::math::Im2ColFunctor<
      paddle::operators::math::ColFormat::kCFO, DeviceContext, float>
      im2col;
  paddle::operators::math::Im2ColFunctor<
      paddle::operators::math::ColFormat::kOCF, DeviceContext, float>
      im2col_ocf;

  im2col(*context, input, dilation, stride, padding, &output_cfo);
  im2col_ocf(*context, input, dilation, stride, padding, &output_ocf);

  float out_cfo_data[] = {0, 1, 1, 2, 3, 4, 4, 5};
  float out_ocf_data[] = {0, 1, 3, 4, 1, 2, 4, 5};

  float* out_cfo_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    out_cfo_ptr = output_cfo.data<float>();
  } else {
    paddle::framework::TensorCopySync(output_cfo, paddle::platform::CPUPlace(),
                                      &output_tmp);
    out_cfo_ptr = output_tmp.data<float>();
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(out_cfo_ptr[i], out_cfo_data[i]);
  }

  float* out_ocf_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    out_ocf_ptr = output_ocf.data<float>();
  } else {
    paddle::framework::TensorCopySync(output_ocf, paddle::platform::CPUPlace(),
                                      &output_tmp);
    out_ocf_ptr = output_tmp.data<float>();
  }

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(out_ocf_ptr[i], out_ocf_data[i]);
  }

  // Col2Im: kCFO
  paddle::operators::math::Col2ImFunctor<
      paddle::operators::math::ColFormat::kCFO, DeviceContext, float>
      col2im;
  paddle::operators::math::Col2ImFunctor<
      paddle::operators::math::ColFormat::kOCF, DeviceContext, float>
      col2im_ocf;
  float col2im_data[] = {0, 2, 2, 3, 8, 5};

  memset(input_ptr, 0, 6 * sizeof(float));
  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }

  col2im(*context, output_cfo, dilation, stride, padding, &input);

  float* in_ptr;
  if (paddle::platform::is_cpu_place(*place)) {
    in_ptr = input.data<float>();
  } else {
    paddle::framework::TensorCopySync(input, paddle::platform::CPUPlace(),
                                      &input_tmp);
    in_ptr = input_tmp.data<float>();
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(in_ptr[i], col2im_data[i]);
  }

  // Col2Im: kOCF
  memset(input_ptr, 0, 6 * sizeof(float));
  if (paddle::platform::is_cpu_place(*place)) {
    input = input_tmp;
  } else {
    paddle::framework::TensorCopySync(input_tmp, *place, &input);
  }

  col2im_ocf(*context, output_ocf, dilation, stride, padding, &input);

  if (paddle::platform::is_cpu_place(*place)) {
    in_ptr = input.data<float>();
  } else {
    paddle::framework::TensorCopySync(input, paddle::platform::CPUPlace(),
                                      &input_tmp);
    in_ptr = input_tmp.data<float>();
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(in_ptr[i], col2im_data[i]);
  }

  delete place;
  delete context;
}

TEST(math, im2col) {
  testIm2col<paddle::platform::CPUDeviceContext, paddle::platform::CPUPlace>();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  testIm2col<paddle::platform::CUDADeviceContext,
             paddle::platform::CUDAPlace>();
#endif
}

#define PREPARE_IM2COL_CPU                                                   \
  paddle::platform::CPUPlace place;                                          \
  paddle::platform::CPUDeviceContext context(place);                         \
  paddle::framework::Tensor input;                                           \
  paddle::framework::Tensor out;                                             \
  paddle::framework::Tensor ref;                                             \
  std::vector<int> padding({ph, pw});                                        \
  std::vector<int> stride({1, 1});                                           \
  std::vector<int> dilation({1, 1});                                         \
  float* input_ptr = input.mutable_data<float>({ic, ih, iw}, place);         \
  for (int i = 0; i < input.numel(); ++i) {                                  \
    input_ptr[i] = static_cast<float>(i + 1);                                \
  }                                                                          \
  int output_height = (ih - fh + padding[0] * 2) / stride[0] + 1;            \
  int output_width = (iw - fw + padding[1] * 2) / stride[1] + 1;             \
  out.mutable_data<float>({ic, fh, fw, output_height, output_width}, place); \
  ref.mutable_data<float>({ic, fh, fw, output_height, output_width}, place); \
  paddle::operators::math::Im2ColFunctor<                                    \
      paddle::operators::math::ColFormat::kCFO,                              \
      paddle::platform::CPUDeviceContext, float>                             \
      im2col

void testIm2colCPU(int ic, int ih, int iw, int fh, int fw, int ph, int pw) {
  PREPARE_IM2COL_CPU;

  im2col(context, input, dilation, stride, padding, &out);
  paddle::operators::math::im2col_common<float>(input, dilation, stride,
                                                padding, &ref);

  float* ref_data = ref.data<float>();
  float* out_data = out.data<float>();
  for (int i = 0; i < out.numel(); ++i) {
    EXPECT_EQ(out_data[i], ref_data[i]);
  }
}

void benchIm2col(int ic, int ih, int iw, int fh, int fw, int ph, int pw) {
  PREPARE_IM2COL_CPU;
  constexpr int repeat = 100;
  auto GetCurrentMs = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
  };
  auto t1 = GetCurrentMs();
  for (int i = 0; i < repeat; ++i) {
    im2col(context, input, dilation, stride, padding, &out);
  }
  auto t2 = GetCurrentMs();

  for (int i = 0; i < repeat; ++i) {
    paddle::operators::math::im2col_common<float>(input, dilation, stride,
                                                  padding, &ref);
  }
  auto t3 = GetCurrentMs();

  LOG(INFO) << "before: " << (t3 - t2) / repeat
            << ",after: " << (t2 - t1) / repeat
            << ",boost: " << ((t3 - t2) / (t2 - t1) - 1) * 100 << "%";
}

TEST(math, im2col_cputest) {
  // padding_h == padding_w
  for (int p = 0; p < 4; ++p) {
    // width == height
    testIm2colCPU(/*ic*/ 2, /*ih*/ 5, /*iw*/ 5, /*fh*/ 4, /*fw*/ 4, /*ph*/ p,
                  /*pw*/ p);
    testIm2colCPU(/*ic*/ 2, /*ih*/ 4, /*iw*/ 4, /*fh*/ 3, /*fw*/ 3, /*ph*/ p,
                  /*pw*/ p);
    testIm2colCPU(/*ic*/ 2, /*ih*/ 4, /*iw*/ 4, /*fh*/ 2, /*fw*/ 2, /*ph*/ p,
                  /*pw*/ p);

    // height != width
    testIm2colCPU(/*ic*/ 2, /*ih*/ 5, /*iw*/ 4, /*fh*/ 2, /*fw*/ 3, /*ph*/ p,
                  /*pw*/ p);
    testIm2colCPU(/*ic*/ 2, /*ih*/ 5, /*iw*/ 4, /*fh*/ 1, /*fw*/ 3, /*ph*/ p,
                  /*pw*/ p);
    testIm2colCPU(/*ic*/ 2, /*ih*/ 4, /*iw*/ 5, /*fh*/ 3, /*fw*/ 1, /*ph*/ p,
                  /*pw*/ p);

    // filter == 1
    testIm2colCPU(/*ic*/ 3, /*ih*/ 4, /*iw*/ 4, /*fh*/ 1, /*fw*/ 1, /*ph*/ p,
                  /*pw*/ p);
    testIm2colCPU(/*ic*/ 3, /*ih*/ 3, /*iw*/ 4, /*fh*/ 1, /*fw*/ 1, /*ph*/ p,
                  /*pw*/ p);
  }

  // padding_h != padding_w
  testIm2colCPU(/*ic*/ 2, /*ih*/ 4, /*iw*/ 4, /*fh*/ 2, /*fw*/ 3, /*ph*/ 1,
                /*pw*/ 2);

  // benchmark
  for (int p : {0, 1}) {
    for (int k : {1, 3, 5}) {
      LOG(INFO) << "padding == " << p << ", filter == " << k;
      benchIm2col(/*ic*/ 3, /*ih*/ 224, /*iw*/ 224, /*fh*/ k, /*fw*/ k,
                  /*ph*/ p, /*pw*/ p);
    }
  }
}
