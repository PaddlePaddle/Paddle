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

__kernel void batchnorm(__private const int out_width,
                        __read_only image2d_t input,
                        __read_only image2d_t new_scale_image,
                        __read_only image2d_t new_bias_image,
                        __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  float4 new_scale = read_imagef(new_scale_image, sampler, (int2)(out_c, 0));
  float4 new_bias = read_imagef(new_bias_image, sampler, (int2)(out_c, 0));

  int pos_x = mad24(out_c, out_width, out_w);
  float4 in = read_imagef(input, sampler, (int2)(pos_x, out_nh));
  float4 out = mad(in, new_scale, new_bias);

  write_imagef(output, (int2)(pos_x, out_nh), out);
}
