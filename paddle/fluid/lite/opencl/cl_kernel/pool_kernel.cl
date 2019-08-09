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

#define MIN_VALUE -FLT_MAX

__kernel void pool_max(
    __private const int in_height, __private const int in_width,
    __private const int out_height, __private const int out_width,
    __private const int pad_top, __private const int pad_left,
    __private const int stride_h, __private const int stride_w,
    __private const int ksize_h, __private const int ksize_w,
    __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h,0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w,0);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  float4 max_value = (float4)(MIN_VALUE);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      float4 tmp = read_imagef(input, sampler, (int2)(pos_in_x + x, pos_in_y + y));
      max_value = max(max_value, tmp);
    }
  }

  const int pos_out_x = mad24(out_c, out_width, out_w);
  write_imagef(output, (int2)(pos_out_x, out_nh), max_value);
}

__kernel void pool_avg(
    __private const int in_height, __private const int in_width,
    __private const int out_height, __private const int out_width,
    __private const int pad_top, __private const int pad_left,
    __private const int stride_h, __private const int stride_w,
    __private const int ksize_h, __private const int ksize_w,
    __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int start_h = max(out_h * stride_h - pad_top, 0);
  int end_h = min(start_h + ksize_h, in_height);

  int start_w = max(out_w * stride_w - pad_left, 0);
  int end_w = min(start_w + ksize_w, in_width);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  float4 sum = (float4)(0.0f);
  int num = 0;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      sum += read_imagef(input, sampler, (int2)(pos_in_x + x, pos_in_y + y));
      num++;
    }
  }
  float4 avg = sum / num;
  const int pos_out_x = mad24(out_c, out_width, out_w);
  write_imagef(output, (int2)(pos_out_x, out_nh), avg);
}
