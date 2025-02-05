// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/runtime/cpu/onednn_math.h"

#include <vector>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/common/enforce.h"

using dnnl::algorithm;
using dnnl::memory;
using tag = memory::format_tag;
using dt = memory::data_type;

void cinn_cpu_onednn_softmax_fp32(int batch,
                                  int channel,
                                  int h,
                                  int w,
                                  int axis,
                                  cinn_buffer_t* inputs,
                                  cinn_buffer_t* out) {
  auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(engine);

  memory::dims src_dims = {batch, channel};
  if (h != 1) src_dims.push_back(h);
  if (w != 1) src_dims.push_back(w);
  int size = src_dims.size();
  auto format_tag = tag::nc;
  switch (size) {
    case 2:
      format_tag = tag::ab;
      break;
    case 3:
      format_tag = tag::abc;
      break;
    case 4:
      format_tag = tag::abcd;
      break;
    default:
      std::stringstream ss;
      ss << "wrong dim: " << size;
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      break;
  }

  auto src_md = memory::desc(src_dims, dt::f32, format_tag);
  auto src_mem =
      memory(src_md, engine, reinterpret_cast<float*>(inputs->memory));
  auto dst_mem = memory(src_md, engine, reinterpret_cast<float*>(out->memory));
  auto softmax_pd =
      dnnl::softmax_forward::primitive_desc(engine,
                                            dnnl::prop_kind::forward_inference,
                                            dnnl::algorithm::softmax_accurate,
                                            src_md,
                                            src_md,
                                            axis);
  auto softmax_prim = dnnl::softmax_forward(softmax_pd);

  softmax_prim.execute(engine_stream,
                       {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  engine_stream.wait();
}

void cinn_cpu_onednn_conv2d_nchw_fp32(int batch_size,
                                      int c_in,
                                      int input_h,
                                      int input_w,
                                      int c_out,
                                      int group,
                                      int filter_h,
                                      int filter_w,
                                      int pad_h,
                                      int pad_w,
                                      int stride_h,
                                      int stride_w,
                                      int dilation_h,
                                      int dilation_w,
                                      cinn_buffer_t* inputs,
                                      cinn_buffer_t* weights,
                                      cinn_buffer_t* out) {
  auto cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream cpu_stream(cpu_engine);

  memory::dims conv_src_tz = {batch_size, c_in, input_h, input_w};
  memory::dims conv_weights_tz = {c_out, c_in, filter_h, filter_w};
  if (group > 1) {
    conv_weights_tz = {group, c_out / group, c_in / group, filter_h, filter_w};
  }
  int out_h =
      (input_h - ((filter_h - 1) * dilation_h + 1) + 2 * pad_h) / stride_h + 1;
  int out_w =
      (input_w - ((filter_w - 1) * dilation_w + 1) + 2 * pad_w) / stride_w + 1;
  memory::dims conv_dst_tz = {batch_size, c_out, out_h, out_w};
  memory::dims conv_strides = {stride_h, stride_w};
  memory::dims conv_paddings = {pad_h, pad_w};
  memory::dims conv_dilations = {dilation_h - 1, dilation_w - 1};

  auto conv_user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw},
                                     cpu_engine,
                                     reinterpret_cast<float*>(inputs->memory));
  auto conv_user_weights_memory =
      memory({{conv_weights_tz}, dt::f32, group > 1 ? tag::goihw : tag::oihw},
             cpu_engine,
             reinterpret_cast<float*>(weights->memory));
  auto conv_user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw},
                                     cpu_engine,
                                     reinterpret_cast<float*>(out->memory));

  auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
  auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
  auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::nchw);

  auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(
      cpu_engine,
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::convolution_direct,
      conv_src_md,
      conv_weights_md,
      conv_dst_md,
      conv_strides,
      conv_dilations,
      conv_paddings,
      conv_paddings);

  auto conv_src_memory = conv_user_src_memory;
  auto conv_weights_memory = conv_user_weights_memory;
  auto conv_dst_memory = conv_user_dst_memory;
  if (conv_prim_desc.dst_desc() != conv_user_dst_memory.get_desc()) {
    conv_dst_memory = memory(conv_prim_desc.dst_desc(), cpu_engine);
  }
  auto conv = dnnl::convolution_forward(conv_prim_desc);
  conv.execute(cpu_stream,
               {{DNNL_ARG_SRC, conv_src_memory},
                {DNNL_ARG_WEIGHTS, conv_weights_memory},
                {DNNL_ARG_DST, conv_dst_memory}});
  if (conv_prim_desc.dst_desc() != conv_user_dst_memory.get_desc()) {
    dnnl::reorder(conv_dst_memory, conv_user_dst_memory)
        .execute(cpu_stream, conv_dst_memory, conv_user_dst_memory);
  } else {
    conv_user_dst_memory = conv_dst_memory;
  }

  cpu_stream.wait();
}

CINN_REGISTER_HELPER(cinn_cpu_onednn) {
  using namespace cinn;  // NOLINT
  using backends::FunctionProto;
  auto host_target = cinn::common::DefaultHostTarget();

  FunctionProto::shape_inference_t inference_shape_conv2d_nchw =
      [](const std::vector<Expr>& args, int offset) {
        PADDLE_ENFORCE_EQ(args.size(),
                          16UL,
                          ::common::errors::InvalidArgument(
                              "Wrong number of arguments passed in."));
        auto N = cinn::optim::ArithSimplify(args[0]);
        int input_h = cinn::optim::ArithSimplify(args[2]).as_int32();
        int input_w = cinn::optim::ArithSimplify(args[3]).as_int32();
        auto c_out = cinn::optim::ArithSimplify(args[4]);
        int filter_h = cinn::optim::ArithSimplify(args[6]).as_int32();
        int filter_w = cinn::optim::ArithSimplify(args[7]).as_int32();
        int pad_h = cinn::optim::ArithSimplify(args[8]).as_int32();
        int pad_w = cinn::optim::ArithSimplify(args[9]).as_int32();
        int stride_h = cinn::optim::ArithSimplify(args[10]).as_int32();
        int stride_w = cinn::optim::ArithSimplify(args[11]).as_int32();
        int dilation_h = cinn::optim::ArithSimplify(args[12]).as_int32();
        int dilation_w = cinn::optim::ArithSimplify(args[13]).as_int32();
        int out_h = (input_h - ((filter_h - 1) * dilation_h + 1) + 2 * pad_h) /
                        stride_h +
                    1;
        int out_w = (input_w - ((filter_w - 1) * dilation_w + 1) + 2 * pad_w) /
                        stride_w +
                    1;

        std::vector<Expr> shape;
        shape.push_back(N);
        shape.push_back(c_out);
        shape.push_back(Expr(out_h));
        shape.push_back(Expr(out_w));
        return shape;
      };

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_onednn_conv2d_nchw_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<int>()              // batch_size
      .AddInputType<int>()              // c_in
      .AddInputType<int>()              // input_h
      .AddInputType<int>()              // input_w
      .AddInputType<int>()              // c_out
      .AddInputType<int>()              // group
      .AddInputType<int>()              // filter_h
      .AddInputType<int>()              // filter_w
      .AddInputType<int>()              // pad_h
      .AddInputType<int>()              // pad_w
      .AddInputType<int>()              // stride_h
      .AddInputType<int>()              // stride_w
      .AddInputType<int>()              // dilation_h
      .AddInputType<int>()              // dilation_w
      .AddInputType<cinn_buffer_t*>()   // inputs
      .AddInputType<cinn_buffer_t*>()   // weights
      .AddOutputType<cinn_buffer_t*>()  // out
      .SetShapeInference(inference_shape_conv2d_nchw)
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_onednn_softmax_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<int>()              // batch_size
      .AddInputType<int>()              // c_in
      .AddInputType<int>()              // h
      .AddInputType<int>()              // w
      .AddInputType<int>()              // axis
      .AddInputType<cinn_buffer_t*>()   // inputs
      .AddOutputType<cinn_buffer_t*>()  // out
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(5))
      .End();

  return true;
}
