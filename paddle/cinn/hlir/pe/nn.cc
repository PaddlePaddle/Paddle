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

#include "paddle/cinn/hlir/pe/nn.h"

#include <absl/container/flat_hash_map.h>

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/nn_util.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Max;
using ir::Min;
using ir::Select;
using ir::Tensor;

std::string Type2StrForNN(common::Type type) {
  std::string suffix;
  if (type.is_float(64)) {
    return "fp64";
  } else if (type.is_float(32)) {
    return "fp32";
  } else if (type.is_bfloat16()) {
    return "bf16";
  } else if (type.is_float16()) {
    return "fp16";
  }
  LOG(FATAL) << "NN Not Support " << type;
  return "";
}

ir::Tensor Relu(const ir::Tensor &A,
                double threshold,
                const std::string &output_name) {
  return lang::Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) {
        return lang::Relu(A(indice), threshold);
      },
      output_name);
}

ir::Tensor Relu6(const ir::Tensor &A,
                 double threshold,
                 const std::string &output_name) {
  return lang::Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) {
        return lang::Relu6(A(indice), threshold);
      },
      output_name);
}

Tensor LeakyRelu(const Tensor &A,
                 double alpha,
                 const std::string &output_name) {
  return Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) {
        return lang::LeakyRelu(A(indice), alpha);
      },
      output_name);
}

Tensor PRelu(const Tensor &A,
             const Tensor &slope,
             const int axis,
             const std::string &output_name) {
  CHECK_LT(axis, A->shape.size()) << "Wrong axis value: " << axis << std::endl;
  CHECK(A->shape[axis] == slope->shape[0])
      << "Wrong slope shape: " << slope->shape[0] << std::endl;
  return Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) {
        return lang::LeakyRelu(A(indice), slope(indice[axis]));
      },
      output_name);
}

std::vector<ir::Tensor> Conv2d_winograd_NCHW(const ir::Tensor &input,
                                             const ir::Tensor &weights,
                                             int pad_h,
                                             int pad_w,
                                             int stride_h,
                                             int stride_w,
                                             int dilation_h,
                                             int dilation_w,
                                             const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Conv2d_winograd_NCHW op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U)
      << "Weight's dimension of Conv2d_winograd_NCHW op is not 4! Please "
         "check.";
  std::vector<Expr> output_shape;
  std::vector<Expr> new_weights_shape;
  std::vector<Expr> input_pad_shape;

  int tile_size = input->shape[2].as_int32() % 8 == 0 ? 4 : 2;

  new_weights_shape = {weights->shape[0],
                       weights->shape[1],
                       dilation_h * (weights->shape[2] - 1) + 1,
                       dilation_w * (weights->shape[3] - 1) + 1};

  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond =
            lang::logic_and({(yy) % dilation_h == 0, xx % dilation_w == 0});
        return ir::Select::Make(
            cond,
            weights(nn, cc, (yy / dilation_h), (xx / dilation_w)),
            common::make_const(weights->type(), 0));
      },
      UniqName("weights_dilation"));

  CHECK(MathEqual((weights->shape[0] * weights->shape[1]) % input->shape[1],
                  Expr(0)))
      << "filter's output channel size must be divisible by group\n";

  int alpha = weights_dilation->shape[3].as_int32() + tile_size - 1;

  input_pad_shape = {input->shape[0],
                     input->shape[1],
                     input->shape[2] + 2 * pad_h,
                     input->shape[3] + 2 * pad_w};

  ir::Tensor input_pad;
  if (pad_h == 0 && pad_w == 0) {
    input_pad = Compute(
        input->shape,
        [=](Expr nn, Expr cc, Expr yy, Expr xx) {
          return input(nn, cc, yy, xx);
        },
        UniqName("input_pad"));
  } else {
    input_pad = Compute(
        input_pad_shape,
        [=](Expr nn, Expr cc, Expr yy, Expr xx) {
          auto cond = lang::logic_and({yy >= pad_h,
                                       yy < input->shape[2] + pad_h,
                                       xx >= pad_w,
                                       xx < input->shape[3] + pad_w});
          return ir::Select::Make(cond,
                                  input(nn, cc, yy - pad_h, xx - pad_w),
                                  ir::Zero(input->type()));
        },
        UniqName("input_pad"));
  }

  int r = weights_dilation->shape[3].as_int32();
  int m = tile_size;

  // # output_shape
  output_shape = {
      input->shape[0],    // B
      weights->shape[0],  // O
      common::AutoSimplify(
          (input->shape[2] -
           ((weights_dilation->shape[2] - 1) * dilation_h + 1) + 2 * pad_h) /
              stride_h +
          1),  // H
      common::AutoSimplify(
          (input->shape[3] -
           ((weights_dilation->shape[3] - 1) * dilation_w + 1) + 2 * pad_w) /
              stride_w +
          1)  // W
  };

  std::vector<ir::Tensor> winograd_transform =
      winograd_transform_matrices(m, r);
  ir::Tensor A = winograd_transform[0];
  ir::Tensor B = winograd_transform[1];
  ir::Tensor G = winograd_transform[2];

  int nH = (common::AutoSimplify(output_shape[2]).as_int32() + m - 1) / m;
  int nW = (common::AutoSimplify(output_shape[3]).as_int32() + m - 1) / m;

  int P = input->shape[0].as_int32() * nH * nW;

  Var r_kh(weights_dilation->shape[2], UniqName("r_kh"));
  Var r_kw(weights_dilation->shape[3], UniqName("r_kw"));
  std::vector<Expr> kernel_shape = {Expr(alpha),
                                    Expr(alpha),
                                    weights_dilation->shape[1],
                                    weights_dilation->shape[0]};
  auto kernel_pack = Compute(
      kernel_shape,
      [=](Expr eps, Expr nu, Expr ci, Expr co) {
        return lang::ReduceSum(
            weights_dilation(co, ci, r_kh, r_kw) * G(eps, r_kh) * G(nu, r_kw),
            {r_kh, r_kw});
      },
      UniqName("kernel_pack"));

  // pack input tile
  std::vector<Expr> input_tile_shape = {
      weights_dilation->shape[1], Expr(P), Expr(alpha), Expr(alpha)};
  auto input_tile = Compute(
      input_tile_shape,
      [=](Expr c, Expr p, Expr eps, Expr nu) {
        return input_pad(
            (p / (nH * nW)), c, ((p / nW) % nH) * m + eps, (p % nW) * m + nu);
      },
      UniqName("input_tile"));

  std::vector<Expr> data_pack_shape = {
      Expr(alpha), Expr(alpha), weights_dilation->shape[1], Expr(P)};
  Var r_a(input_tile->shape[2], UniqName("r_a"));
  Var r_b(input_tile->shape[3], UniqName("r_b"));
  auto data_pack = Compute(
      data_pack_shape,
      [=](Expr eps, Expr nu, Expr ci, Expr p) {
        return lang::ReduceSum(
            input_tile(ci, p, r_a, r_b) * B(r_a, eps) * B(r_b, nu), {r_a, r_b});
      },
      UniqName("data_pack"));

  // do batch gemm
  std::vector<Expr> bgemm_shape = {
      Expr(alpha), Expr(alpha), weights_dilation->shape[0], Expr(P)};
  Var ci(kernel_pack->shape[2], UniqName("ci"));
  auto bgemm = Compute(
      bgemm_shape,
      [=](Expr eps, Expr nu, Expr co, Expr p) {
        return lang::ReduceSum(
            kernel_pack(eps, nu, ci, co) * data_pack(eps, nu, ci, p), {ci});
      },
      UniqName("bgemm"));

  // # inverse transform
  std::vector<Expr> inverse_shape = {
      weights_dilation->shape[0], Expr(P), Expr(m), Expr(m)};
  Var r_g_a(bgemm->shape[0], UniqName("r_g_a"));
  Var r_g_b(bgemm->shape[1], UniqName("r_g_b"));
  auto inverse = Compute(
      inverse_shape,
      [=](Expr co, Expr p, Expr vh, Expr vw) {
        return lang::ReduceSum(
            bgemm(r_g_a, r_g_b, co, p) * A(r_g_a, vh) * A(r_g_b, vw),
            {r_g_a, r_g_b});
      },
      UniqName("inverse"));
  auto res = Compute(
      output_shape,
      [=](Expr n, Expr co, Expr h, Expr w) {
        return inverse(
            co, n * nH * nW + (h / m) * nW + (w / m), (h % m), (w % m));
      },
      output_name);

  return {weights_dilation,
          input_pad,
          A,
          B,
          G,
          kernel_pack,
          input_tile,
          data_pack,
          bgemm,
          inverse,
          res};
}

std::vector<ir::Tensor> Conv2d_NCHW(const ir::Tensor &input,
                                    const ir::Tensor &weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    const std::string &output_name,
                                    bool choose_direct_compute) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Conv2d_NCHW op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U)
      << "Weight's dimension of Conv2d_NCHW op is not 4! Please check.";
  std::vector<int> output_shape_int;
  std::vector<int> new_weights_shape_int;
  std::vector<int> input_pad_shape_int;
  output_shape_int = {
      input->shape[0].as_int32(),    // B
      weights->shape[0].as_int32(),  // O
      (input->shape[2].as_int32() -
       ((weights->shape[2].as_int32() - 1) * dilation_h + 1) + 2 * pad_h) /
              stride_h +
          1,  // H
      (input->shape[3].as_int32() -
       ((weights->shape[3].as_int32() - 1) * dilation_w + 1) + 2 * pad_w) /
              stride_w +
          1  // W
  };
  new_weights_shape_int = {weights->shape[0].as_int32(),
                           weights->shape[1].as_int32(),
                           dilation_h * (weights->shape[2].as_int32() - 1) + 1,
                           dilation_w * (weights->shape[3].as_int32() - 1) + 1};
  input_pad_shape_int = {input->shape[0].as_int32(),
                         input->shape[1].as_int32(),
                         input->shape[2].as_int32() + 2 * pad_h,
                         input->shape[3].as_int32() + 2 * pad_w};
  std::vector<Expr> output_shape{Expr(output_shape_int[0]),
                                 Expr(output_shape_int[1]),
                                 Expr(output_shape_int[2]),
                                 Expr(output_shape_int[3])};
  std::vector<Expr> new_weights_shape{Expr(new_weights_shape_int[0]),
                                      Expr(new_weights_shape_int[1]),
                                      Expr(new_weights_shape_int[2]),
                                      Expr(new_weights_shape_int[3])};
  std::vector<Expr> input_pad_shape{Expr(input_pad_shape_int[0]),
                                    Expr(input_pad_shape_int[1]),
                                    Expr(input_pad_shape_int[2]),
                                    Expr(input_pad_shape_int[3])};
  CHECK_EQ(weights->shape.size(), 4);
  CHECK(weights->shape[2].is_constant());
  CHECK(weights->shape[3].is_constant());
  int kh = weights->shape[2].as_int32();
  int kw = weights->shape[3].as_int32();
  if (!choose_direct_compute && stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1 && 2 < kh && kh < 8 && 2 < kw &&
      kw < 8) {
    auto &res = ScheduleParam::get_cuda_instance().GetParam();
    std::string key = "CudaWinogradConvSchedule " +
                      std::to_string(input_pad_shape_int[0]) + " " +
                      std::to_string(input_pad_shape_int[1]) + " " +
                      std::to_string(input_pad_shape_int[2]) + " " +
                      std::to_string(input_pad_shape_int[3]) + " " +
                      std::to_string(new_weights_shape_int[0]) + " " +
                      std::to_string(new_weights_shape_int[1]) + " " +
                      std::to_string(new_weights_shape_int[2]) + " " +
                      std::to_string(new_weights_shape_int[3]) + " " +
                      std::to_string(output_shape_int[0]) + " " +
                      std::to_string(output_shape_int[1]) + " " +
                      std::to_string(output_shape_int[2]) + " " +
                      std::to_string(output_shape_int[3]);
    if (res.count(key) > 0) {
      VLOG(3) << "Find saved winograd_conv2d schedule param! key is: " << key;
      return Conv2d_winograd_NCHW(input,
                                  weights,
                                  pad_h,
                                  pad_w,
                                  stride_h,
                                  stride_w,
                                  dilation_h,
                                  dilation_w,
                                  output_name);
    }
    VLOG(3) << "Didn't find saved winograd_conv2d schedule param! key is: "
            << key;
  }
  ir::Tensor input_pad;
  if (pad_h == 0 && pad_w == 0) {
    input_pad = Compute(
        input->shape,
        [=](Expr nn, Expr cc, Expr yy, Expr xx) {
          return input(nn, cc, yy, xx);
        },
        UniqName("input_pad"));
  } else {
    input_pad = Compute(
        input_pad_shape,
        [=](Expr nn, Expr cc, Expr yy, Expr xx) {
          auto cond = lang::logic_and({yy >= pad_h,
                                       yy < input->shape[2] + pad_h,
                                       xx >= pad_w,
                                       xx < input->shape[3] + pad_w});
          return ir::Select::Make(cond,
                                  input(nn, cc, yy - pad_h, xx - pad_w),
                                  ir::Zero(input->type()));
        },
        UniqName("input_pad"));
  }

  Var rc(weights->shape[1], UniqName("rc"));
  Var ry(weights->shape[2], UniqName("ry"));
  Var rx(weights->shape[3], UniqName("rx"));

  CHECK(MathEqual((weights->shape[0] * weights->shape[1]) % input->shape[1],
                  Expr(0)))
      << "filter's output channel size must be divisible by group\n";
  auto res = Compute(
      output_shape,
      [=](Expr nn, Expr ff, Expr yy, Expr xx) {
        return lang::ReduceSum(input_pad(nn,
                                         rc,
                                         yy * stride_h + ry * dilation_h,
                                         xx * stride_w + rx * dilation_w) *
                                   weights(ff, rc, ry, rx),
                               {rc, ry, rx});
      },
      output_name);
  return {res, input_pad};
}

std::vector<ir::Tensor> Conv2d_NCHW_5D(const ir::Tensor &input,
                                       const ir::Tensor &weights,
                                       int pad_h,
                                       int pad_w,
                                       int stride_h,
                                       int stride_w,
                                       int dilation_h,
                                       int dilation_w,
                                       std::string key,
                                       const std::string &output_name,
                                       const common::Target &target) {
  // input: 4D to 5D, NCHW->NCHWc
  // [batch, in_channel, in_height, in_width] ->
  // [batch, in_channel_chunk, in_height, in_width, in_channel_block]
  auto type = input->type();
  std::vector<Expr> shape_input = input->shape;
  std::vector<Expr> shape_weights = weights->shape;
  CHECK_EQ(shape_input.size(), 4U) << "input's shape size should be 4";
  CHECK_EQ(shape_weights.size(), 4U) << "weight's shape size should be 4";
  Expr c_in = common::AutoSimplify(shape_input[1]);
  Expr c_filter = common::AutoSimplify(shape_weights[1]);
  Expr c_out = common::AutoSimplify(shape_weights[0]);
  absl::flat_hash_map<std::string, int> conv2d_factors;
  int oc = c_out.as_int32();
  int ic = c_in.as_int32();
  int fc_size = c_filter.as_int32();
  if (key.empty()) {
    key = GenerateX86ConvKey(shape_input,
                             shape_weights,
                             {stride_h, stride_w},
                             {pad_h, pad_w},
                             {dilation_h, dilation_w});
  }
  GetConv2dFactors(&conv2d_factors, oc, ic, fc_size, -1, -1, type, target, key);
  int ic_bn_size = conv2d_factors["ic_bn"];
  int oc_bn_size = conv2d_factors["oc_bn"];
  int fc_bn_size = conv2d_factors["fc_bn"];
  VLOG(3) << "oc_bn: " << oc_bn_size;
  VLOG(3) << "ic_bn: " << ic_bn_size;
  VLOG(3) << "fc_bn: " << fc_bn_size;
  Expr ic_bn = Expr(ic_bn_size);
  Expr oc_bn = Expr(oc_bn_size);
  Expr fc_bn = Expr(fc_bn_size);
  Expr ic_chunk = c_in / ic_bn;
  Expr oc_chunk = c_out / oc_bn;
  Expr fc_chunk = c_filter / fc_bn;

  // pack data, 4D->5D
  Expr batch = shape_input[0];
  Expr h_in = shape_input[2];
  Expr w_in = shape_input[3];
  Expr h_f = shape_weights[2];
  Expr w_f = shape_weights[3];
  auto data = Compute(
      {batch, ic_chunk, h_in, w_in, ic_bn},
      [=](Expr n, Expr icc, Expr h, Expr w, Expr icb) {
        return input(n, icc * ic_bn + icb, h, w);
      },
      UniqName("data_vec"));
  // pack kernel, 4D->6D
  std::vector<Expr> new_weights_shape;
  new_weights_shape = {
      oc_chunk, fc_chunk, shape_weights[2], shape_weights[3], fc_bn, oc_bn};

  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr occ, Expr fcc, Expr yy, Expr xx, Expr fcb, Expr ocb) {
        return weights(occ * oc_bn + ocb, fcc * ic_bn + fcb, yy, xx);
      },
      UniqName("weights_dilation_vec"));

  auto tensors = Conv2d_NCHWc(data,
                              weights_dilation,
                              pad_h,
                              pad_w,
                              stride_h,
                              stride_w,
                              dilation_h,
                              dilation_w);
  CHECK_EQ(tensors.size(), 2U) << "Conv2d_NCHWc should return 2 tensors";
  auto packed_out = tensors[0];
  auto input_pad = tensors[1];
  // 5D back to 4D, NCHWc->NCHW
  std::vector<Expr> output_shape = {
      batch,  // B
      c_out,  // O
      common::AutoSimplify((h_in - ((h_f - 1) * dilation_h + 1) + 2 * pad_h) /
                               stride_h +
                           1),  // H
      common::AutoSimplify((w_in - ((w_f - 1) * dilation_w + 1) + 2 * pad_w) /
                               stride_w +
                           1)  // W
  };
  auto res = Compute(
      output_shape,
      [=](Expr n, Expr c, Expr h, Expr w) {
        return packed_out(n, c / oc_bn, h, w, c % oc_bn);
      },
      UniqName("conv2d_nchw_out"));
  return {res, packed_out, weights_dilation, input_pad, data};
}

std::vector<ir::Tensor> Conv2d_NCHWc(const ir::Tensor &input,
                                     const ir::Tensor &weights,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int dilation_h,
                                     int dilation_w,
                                     const std::string &output_name,
                                     const common::Target &target) {
  // input: [N, c_in_outer, H, W, c_in_inner]
  // weight: [c_out_outer, c_filter_outer, filter_h, filter_w, c_filter_inner,
  // c_out_inner]
  auto type = input->type();
  std::vector<Expr> shape_input = input->shape;
  std::vector<Expr> shape_weights = weights->shape;
  CHECK_EQ(shape_input.size(), 5U)
      << "Conv2d_NCHWc input's shape size should be 5";
  CHECK_EQ(shape_weights.size(), 6U)
      << "Conv2d_NCHWc weight's shape size should be 6";

  Expr batch = shape_input[0];
  Expr c_in_outer = common::AutoSimplify(shape_input[1]);
  Expr h_in = shape_input[2];
  Expr w_in = shape_input[3];
  Expr c_in_inner = common::AutoSimplify(shape_input[4]);

  Expr c_out_outer = shape_weights[0];
  Expr c_filter_outer = common::AutoSimplify(shape_weights[1]);
  Expr h_f = shape_weights[2];
  Expr w_f = shape_weights[3];
  Expr c_filter_inner = common::AutoSimplify(shape_weights[4]);
  Expr c_out_inner = common::AutoSimplify(shape_weights[5]);

  Expr c_filter = common::AutoSimplify(c_filter_outer * c_filter_inner);
  Expr c_out = common::AutoSimplify(c_out_outer * c_out_inner);
  Expr c_in = common::AutoSimplify(c_in_outer * c_in_inner);
  Var fc(c_filter, UniqName("fc"));
  Var fy(h_f, UniqName("fy"));
  Var fx(w_f, UniqName("fx"));
  std::vector<Expr> output_shape = {
      batch,        // B
      c_out_outer,  // O
      common::AutoSimplify((h_in - ((h_f - 1) * dilation_h + 1) + 2 * pad_h) /
                               stride_h +
                           1),  // H
      common::AutoSimplify((w_in - ((w_f - 1) * dilation_w + 1) + 2 * pad_w) /
                               stride_w +
                           1),  // W
      c_out_inner};

  ir::Tensor input_pad;
  if (pad_h == 0 && pad_w == 0) {
    input_pad = Compute(
        input->shape,
        [=](Expr n, Expr icc, Expr yy, Expr xx, Expr icb) {
          return input(n, icc, yy, xx, icb);
        },
        UniqName("input_pad"));
  } else {
    auto pad_h_bound = common::AutoSimplify((output_shape[2] - 1) * stride_h +
                                            (h_f - 1) * dilation_h + 1);
    auto pad_w_bound = common::AutoSimplify((output_shape[3] - 1) * stride_w +
                                            (w_f - 1) * dilation_w + 1);
    auto pad_out_h =
        std::min(pad_h_bound.as_int32(),
                 common::AutoSimplify(h_in + 2 * pad_h).as_int32());
    auto pad_out_w =
        std::min(pad_w_bound.as_int32(),
                 common::AutoSimplify(w_in + 2 * pad_w).as_int32());
    auto h_in_pad = common::AutoSimplify(h_in + pad_h);
    auto w_in_pad = common::AutoSimplify(w_in + pad_w);
    input_pad = Compute(
        {batch, c_in_outer, Expr(pad_out_h), Expr(pad_out_w), c_in_inner},
        [=](Expr n, Expr icc, Expr yy, Expr xx, Expr icb) {
          auto cond = lang::logic_and({yy >= pad_h, xx >= pad_w});
          if (pad_out_h > h_in_pad.as_int32()) {
            cond = lang::logic_and({cond, yy < h_in_pad});
          }
          if (pad_out_w > w_in_pad.as_int32()) {
            cond = lang::logic_and({cond, xx < w_in_pad});
          }
          return ir::Select::Make(
              cond, input(n, icc, yy - pad_h, xx - pad_w, icb), ir::Zero(type));
        },
        UniqName("input_pad"));
  }

  auto packed_out = Compute(
      output_shape,
      [=](Expr n, Expr oc_chunk, Expr oh, Expr ow, Expr oc_block) {
        Expr c_out_per_group = common::AutoSimplify(c_out * c_filter / c_in);
        Expr ic_outer, ic_inner;
        if (c_in == c_filter) {
          ic_outer = common::AutoSimplify(fc / c_in_inner);
          ic_inner = common::AutoSimplify(fc % c_in_inner);
        } else {
          ic_outer = common::AutoSimplify(((oc_chunk * c_out_inner + oc_block) /
                                               c_out_per_group * c_filter +
                                           fc) /
                                          c_in_inner);
          ic_inner = common::AutoSimplify(((oc_chunk * c_out_inner + oc_block) /
                                               c_out_per_group * c_filter +
                                           fc) %
                                          c_in_inner);
        }
        return lang::ReduceSum(input_pad(n,
                                         ic_outer,
                                         oh * stride_h + fy * dilation_h,
                                         ow * stride_w + fx * dilation_w,
                                         ic_inner) *
                                   weights(oc_chunk,
                                           fc / c_filter_inner,
                                           fy,
                                           fx,
                                           fc % c_filter_inner,
                                           oc_block),
                               {fc, fy, fx});
      },
      UniqName("conv2d_NCHWc_out"));
  return {packed_out, input_pad};
}

#ifdef CINN_WITH_DNNL
std::vector<ir::Tensor> Conv2d_NCHW_MKLDNN(const ir::Tensor &input,
                                           const ir::Tensor &weights,
                                           int pad_h,
                                           int pad_w,
                                           int stride_h,
                                           int stride_w,
                                           int dilation_h,
                                           int dilation_w,
                                           const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Conv2d_NCHW op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U)
      << "Weight's dimension of Conv2d_NCHW op is not 4! Please check.";
  std::vector<Expr> output_shape;
  std::vector<Expr> new_weights_shape;
  std::vector<Expr> input_pad_shape;
  int group = input->shape[1].as_int32() / weights->shape[1].as_int32();
  CHECK_EQ(input->shape[1].as_int32(), weights->shape[1].as_int32() * group)
      << "input channel should be divisible by filter channel";
  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkldnn_conv2d_nchw_fp32",
                                {
                                    Expr(input->shape[0]),    // batch_size
                                    Expr(input->shape[1]),    // c_in
                                    Expr(input->shape[2]),    // input_h
                                    Expr(input->shape[3]),    // input_w
                                    Expr(weights->shape[0]),  // c_out
                                    Expr(group),              // group
                                    Expr(weights->shape[2]),  // filter_h
                                    Expr(weights->shape[3]),  // filter_w
                                    Expr(pad_h),              // pad_h
                                    Expr(pad_w),              // pad_w
                                    Expr(stride_h),           // stride_h
                                    Expr(stride_w),           // stride_w
                                    Expr(dilation_h),         // dilation_h
                                    Expr(dilation_w),         // dilation_w
                                    input,                    // input
                                    weights                   // weights
                                });
      },
      UniqName("conv2d_nchw_mkldnn_out"));
  auto out = call->TupleGet(0);
  out->WithBuffer(input->type());
  return {out, call};
}
#endif

std::vector<ir::Tensor> Conv2d_NHWC(const ir::Tensor &input,
                                    const ir::Tensor &weights,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Conv2d_NHWC op is not 4! Please check.";
  CHECK_EQ(weights->shape.size(), 4U)
      << "Weight's dimension of Conv2d_NHWC op is not 4! Please check.";
  std::vector<Expr> output_shape;
  std::vector<Expr> new_weights_shape;
  std::vector<Expr> input_pad_shape;

  output_shape = {
      input->shape[0],  // B
      Expr((input->shape[1] - ((weights->shape[2] - 1) * dilation_h + 1) +
            2 * pad_h) /
               stride_h +
           1),  // H
      Expr((input->shape[2] - ((weights->shape[3] - 1) * dilation_w + 1) +
            2 * pad_w) /
               stride_w +
           1),           // W
      weights->shape[0]  // O
  };
  new_weights_shape = {weights->shape[0],
                       weights->shape[1],
                       dilation_h * (weights->shape[2] - 1) + 1,
                       dilation_w * (weights->shape[3] - 1) + 1};
  input_pad_shape = {input->shape[0],
                     input->shape[1] + 2 * pad_h,
                     input->shape[2] + 2 * pad_w,
                     input->shape[3]};
  auto input_pad = Compute(
      input_pad_shape,
      [=](Expr nn, Expr yy, Expr xx, Expr cc) {
        auto cond = lang::logic_and({yy >= pad_h,
                                     yy - pad_h < input->shape[1],
                                     xx >= pad_w,
                                     xx - pad_w < input->shape[2]});
        return ir::Select::Make(cond,
                                input(nn, yy - pad_h, xx - pad_w, cc),
                                ir::Zero(input->type()));
      },
      UniqName("input_pad"));

  auto weights_dilation = Compute(
      new_weights_shape,
      [=](Expr nn, Expr cc, Expr yy, Expr xx) {
        auto cond =
            lang::logic_and({(yy) % dilation_h == 0, xx % dilation_w == 0});
        return ir::Select::Make(
            cond,
            weights(nn, cc, yy / dilation_h, xx / dilation_w),
            common::make_const(weights->type(), 0));
      },
      UniqName("weights_dilation"));

  Var fc(weights->shape[1], UniqName("fc"));
  Var fy(weights_dilation->shape[2], UniqName("fy"));
  Var fx(weights_dilation->shape[3], UniqName("fx"));

  CHECK(MathEqual((weights->shape[0] * weights->shape[1]) % input->shape[3],
                  Expr(0)))
      << "filter's output channel size must be divisible by group\n";
  auto res = Compute(
      output_shape,
      [=](Expr nn, Expr yy, Expr xx, Expr ff) {
        return lang::ReduceSum(
            input_pad(
                nn,
                yy * stride_h + fy,
                xx * stride_w + fx,
                ff / (weights->shape[0] * weights->shape[1] / input->shape[3]) *
                        weights->shape[1] +
                    fc) *
                weights_dilation(ff, fc, fy, fx),
            {fy, fx, fc});
      },
      output_name);
  return {res, input_pad, weights_dilation};
}

std::vector<Tensor> Depthwise_Conv2d_NCHW(const Tensor &input,
                                          const Tensor &weight,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          const std::string output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  CHECK_EQ(weight->shape.size(), 4U)
      << "Weight's dimension of Depthwise_Conv2d_NCHW is not 4! Please "
         "check.\n";
  Expr in_h = input->shape[2];
  Expr in_w = input->shape[3];
  Expr c_m = weight->shape[1];  // channel_multiplier
  std::vector<Expr> output_shape;
  CHECK(input->shape[0].is_constant());
  CHECK(input->shape[1].is_constant());
  CHECK(input->shape[2].is_constant());
  CHECK(input->shape[3].is_constant());
  CHECK(weight->shape[1].is_constant());
  CHECK(weight->shape[2].is_constant());
  CHECK(weight->shape[3].is_constant());
  int B = static_cast<int>(input->shape[0].get_constant());
  int O = static_cast<int>(weight->shape[1].get_constant()) *
          static_cast<int>(input->shape[1].get_constant());
  int H = (static_cast<int>(input->shape[2].get_constant()) -
           static_cast<int>(weight->shape[2].get_constant()) + 2 * pad_h) /
              stride_h +
          1;
  int W = (static_cast<int>(input->shape[3].get_constant()) -
           static_cast<int>(weight->shape[3].get_constant()) + 2 * pad_w) /
              stride_w +
          1;
  output_shape = {
      Expr(B),  // B
      Expr(O),  // O
      Expr(H),  // H
      Expr(W)   // W
  };
  auto input_pad =
      (pad_h == 0 && pad_w == 0)
          ? Identity(input).front()
          : Pad(input, {Expr(0), Expr(0), Expr(pad_h), Expr(pad_w)});

  Var kernel_h = Var(weight->shape[2], "kh");
  Var kernel_w = Var(weight->shape[3], "kw");
  VLOG(3) << "Output shape is : " << cinn::utils::Join(output_shape, ",");
  auto res = Compute(
      output_shape,
      [=](Expr nn, Expr ff, Expr yy, Expr xx) {
        return lang::ReduceSum(
            input_pad(nn,
                      ff / c_m,
                      yy * stride_h + kernel_h,
                      xx * stride_w + kernel_w) *
                weight(ff / c_m, ff % c_m, kernel_h, kernel_w),
            {kernel_h, kernel_w});
      },
      output_name);
  return {res, input_pad};
}

std::vector<Tensor> Depthwise_Conv2d_NHWC(const Tensor &input,
                                          const Tensor &weight,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          const std::string output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of Depthwise_Conv2d_NCHW is not 4! Please check.\n";
  CHECK_EQ(weight->shape.size(), 4U)
      << "Weight's dimension of Depthwise_Conv2d_NCHW is not 4! Please "
         "check.\n";
  Expr in_h = input->shape[1];
  Expr in_w = input->shape[2];
  Expr c_m = weight->shape[1];  // channel_multiplier
  std::vector<Expr> output_shape;

  output_shape = {
      input->shape[0],                                                  // B
      (input->shape[1] - weight->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (input->shape[2] - weight->shape[3] + 2 * pad_w) / stride_w + 1,  // W
      weight->shape[1] * input->shape[3]                                // O
  };

  auto input_pad =
      (pad_h == 0 && pad_w == 0)
          ? Identity(input).front()
          : Pad(input, {Expr(0), Expr(pad_h), Expr(pad_w), Expr(0)});

  Var kernel_h = Var(weight->shape[2], "kh");
  Var kernel_w = Var(weight->shape[3], "kw");
  auto res = Compute(
      output_shape,
      [=](Expr nn, Expr yy, Expr xx, Expr ff) {
        return lang::ReduceSum(
            input_pad(nn,
                      yy * stride_h + kernel_h,
                      xx * stride_w + kernel_w,
                      ff / c_m) *
                weight(ff / c_m, ff % c_m, kernel_h, kernel_w),
            {kernel_h, kernel_w});
      },
      output_name);
  return {res, input_pad};
}

/**
 * Can be used as a normalizer function for convolution or fully_connected
 * operations. Specified for NCHW layout. Math: Y = (X - mean) / sqrt(variance +
 * epsilon) * scale + bias
 * @param input The input variable.
 * @param weights The weights containing mean, variance, scale and bias.
 * @param epsilon The param epsilon is added to avoid divide zero.
 * @param output_name The name of output tensor.
 * @return The calculated output tensor.
 */
ir::Tensor BatchNorm_NCHW(const ir::Tensor &input,
                          const ir::Tensor &scale,
                          const ir::Tensor &bias,
                          const ir::Tensor &mean,
                          const ir::Tensor &variance,
                          float epsilon,
                          const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 4U)
      << "Input's dimension of BatchNorm op is not 4! Please check.";
  CHECK_EQ(scale->shape.size(), 1U)
      << "Scale's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(bias->shape.size(), 1U)
      << "Bias's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(mean->shape.size(), 1U)
      << "Mean's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(variance->shape.size(), 1U)
      << "Variance's dimension of BatchNorm op is not 1! Please check.";
  auto res = Compute(
      input->shape,
      [=](Expr n, Expr c, Expr h, Expr w) {
        return (input(n, c, h, w) - mean(c)) * scale(c) /
                   lang::Sqrt(variance(c) +
                              common::make_const(input->type(), epsilon)) +
               bias(c);
      },
      UniqName(output_name));
  return res;
}

ir::Tensor BatchNorm_NCHWc(const ir::Tensor &input,
                           const ir::Tensor &scale,
                           const ir::Tensor &bias,
                           const ir::Tensor &mean,
                           const ir::Tensor &variance,
                           float epsilon,
                           const std::string &output_name) {
  CHECK_EQ(input->shape.size(), 5U)
      << "Input's dimension of BatchNorm op is not 5! Please check.";
  CHECK_EQ(scale->shape.size(), 1U)
      << "Scale's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(bias->shape.size(), 1U)
      << "Bias's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(mean->shape.size(), 1U)
      << "Mean's dimension of BatchNorm op is not 1! Please check.";
  CHECK_EQ(variance->shape.size(), 1U)
      << "Variance's dimension of BatchNorm op is not 1! Please check.";
  Expr ic_bn = input->shape.back();
  auto res = Compute(
      input->shape,
      [=](Expr n, Expr icc, Expr h, Expr w, Expr icb) {
        Expr new_c = icc * ic_bn + icb;
        return (input(n, icc, h, w, icb) - mean(new_c)) * scale(new_c) /
                   lang::Sqrt(variance(new_c) +
                              common::make_const(input->type(), epsilon)) +
               bias(new_c);
      },
      UniqName(output_name));
  return res;
}

/**
 * This operator implements the softmax layer.
 * @param A The input tensor.
 * @param axis The axis parameter.
 * @param output_name The name of output tensor.
 * @return The calculated output tensor.
 */
std::vector<ir::Tensor> Softmax(const ir::Tensor &A,
                                int axis,
                                const std::string &output_name) {
  if (axis == -1) {
    axis = A->shape.size() - 1;
  }
  Var reduce_axis(A->shape[axis], UniqName("reduce_axis"));
  std::vector<Expr> new_shapes;
  for (size_t i = 0; i < A->shape.size(); i++) {
    if (static_cast<int>(i) != axis) {
      new_shapes.push_back(A->shape[i]);
    }
  }
  auto temp = Compute(
      new_shapes,
      [=](const std::vector<Expr> &indice) {
        std::vector<Expr> new_indice;
        int count = 0;
        for (size_t i = 0; i < A->shape.size(); i++) {
          if (static_cast<int>(i) != axis) {
            new_indice.push_back(indice[count++]);
          } else {
            new_indice.push_back(reduce_axis);
          }
        }
        return lang::ReduceSum(lang::Exp(A(new_indice)), {reduce_axis});
      },
      UniqName("softmax_temp_out"));
  temp->WithBuffer("local");

  ir::Tensor out = Compute(
      A->shape,
      [=](const std::vector<Expr> &indice) {
        std::vector<Expr> new_indice;
        for (size_t i = 0; i < indice.size(); i++) {
          if (static_cast<int>(i) != axis) {
            new_indice.push_back(indice[i]);
          }
        }
        return lang::Exp(A(indice)) / temp(new_indice);
      },
      output_name);
  return {out, temp};
}

#ifdef CINN_WITH_DNNL
std::vector<ir::Tensor> SoftmaxMKLDNN(const ir::Tensor &A,
                                      int axis,
                                      const std::string &output_name) {
  CHECK_LE(A->shape.size(), 4U)
      << "Input's dimension of mkldnn softmax op is less than 4! Please check.";
  if (axis == -1) {
    axis = A->shape.size() - 1;
  }
  auto shape = A->shape;
  for (size_t i = shape.size(); i < 4; i++) {
    shape.push_back(Expr(1));
  }

  auto call = Compute(
      {Expr(1)},
      [=]() -> Expr {
        return lang::CallExtern("cinn_cpu_mkldnn_softmax_fp32",
                                {
                                    shape[0],    // batch_size
                                    shape[1],    // c_in
                                    shape[2],    // h
                                    shape[3],    // w
                                    Expr(axis),  // axis
                                    A,           // input
                                });
      },
      output_name);
  auto out = call->TupleGet(0);
  out->WithBuffer(A->type());
  return {out, call};
}
#endif

/**
 * @brief Perform padding operation.
 * @param tensor The input tensor.
 * @param pad_before Vector of Exprs describing the padding before the
 * respective dimension
 * @param pad_after Vector of Exprs describing the padding after the respective
 * dimension
 * @param pad_value The value to fill padding elements with. Default is zero.
 * @param name The name of the output padding tensor
 * @param pad_mode Padding type to use: "constant" pads with constant_value;
 * "edge" pads using the edge values of the input array; "reflect" pads by
 * reflecting values with respect to the edges.
 *
 * @return the output tensor after padding.
 *
 * @note
 *  The pad_after vector must either be empty or have the same length as
 * pad_before When pad_after is empty, it takes the same values as pad_before
 * (symmetric padding) The pad vector applies from the leading dimensions and
 * skips missing trailing dimensions: e.g. pad(t(i, j, k), {1}, {1}) returns the
 * equivalent operation for the following pseudocode: for i in [0, t.shape[0] +
 * 2): for j in [0, t.shape[0] + 2): for k in [0, t.shape[0] + 2): name(i,j,k) =
 *                             i < 1 ? 0 :
 *                               ((1 <= i < t.shape[0] + 1) ?
 *                                 t(i-1, j, k) : 0));
 *
 */
Tensor Pad(const Tensor &tensor,
           const std::vector<Expr> &pad_before,
           std::vector<Expr> pad_after,
           Expr pad_value,
           const std::string &name,
           const std::string &pad_mode) {
  // When pad_after is empty, it takes the same values as pad_before (symmetric
  // padding)
  if (pad_after.size() < pad_before.size()) {
    for (size_t i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }
  CHECK(!pad_before.empty());
  CHECK_EQ(pad_before.size(), pad_after.size());
  std::vector<Expr> output_shape;
  for (auto &ele : pad_before) {
    CHECK(ele.type().is_int(32)) << "padding size should be int32\n";
  }
  for (auto &ele : pad_after) {
    CHECK(ele.type().is_int(32)) << "padding size should be int32\n";
  }
  for (size_t i = 0; i < tensor->shape.size(); ++i) {
    if (i >= pad_before.size()) {
      output_shape.push_back(tensor->shape[i]);
    } else {
      auto shape =
          common::AutoSimplify(tensor->shape[i] + pad_before[i] + pad_after[i]);
      output_shape.push_back(shape);
    }
  }
  // default value is zero
  if (!pad_value.defined()) {
    pad_value = make_const(tensor->type(), 0);
  }

  auto fn = [=](const std::vector<Expr> &ovars) {
    std::vector<Expr> indices;
    std::vector<Expr> sel;
    std::vector<Expr> pad_idx;
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        indices.emplace_back(ovars[i]);
        continue;
      }
      if (!MathEqual(pad_before[i], Expr(0))) {
        sel.push_back(ir::GE::Make(ovars[i], pad_before[i]));
        indices.push_back(ovars[i] - pad_before[i]);
      } else {
        indices.emplace_back(ovars[i]);
      }
      Expr sel_after;
      if (!MathEqual(pad_after[i], Expr(0))) {
        sel_after =
            common::AutoSimplify(ovars[i] < pad_before[i] + tensor->shape[i]);
        sel.push_back(sel_after);
      }
      if (pad_mode == "edge") {
        pad_idx.push_back(Select::Make(
            ovars[i] < pad_before[i],
            0,
            Select::Make(ovars[i] >= pad_before[i] + tensor->shape[i],
                         tensor->shape[i] - 1,
                         ovars[i] - pad_before[i])));
      } else if (pad_mode == "reflect") {
        pad_idx.push_back(Select::Make(
            ovars[i] < pad_before[i],
            pad_before[i] - ovars[i],
            Select::Make(ovars[i] >= pad_before[i] + tensor->shape[i],
                         tensor->shape[i] * 2 - ovars[i] + pad_before[i] - 2,
                         ovars[i] - pad_before[i])));
      }
    }
    if (sel.size() != 0) {
      auto fn = [](Expr a, Expr b) { return a && b; };
      if (pad_mode == "constant") {
        return Select::Make(FoldExpr(fn, sel), tensor(indices), pad_value);
      } else if (pad_mode == "edge" || pad_mode == "reflect") {
        return Select::Make(
            FoldExpr(fn, sel), tensor(indices), tensor(pad_idx));
      }
    }
    return tensor(indices);
  };
  return Compute(output_shape, fn, UniqName(name));
}

/**
 * @brief Perform pooling on N-dimension of data.
 *
 * @param tensor The input tensor with the shape of {N, C, H, W} or {N, H, W,
 * C}.
 * @param kernel_size Vector of N ints that indicates pooling kernel size. If N
 * is 2, then is {pool_kernel_Height, pool_kernel_Width}.
 * @param stride_size Vector of N ints that indicates pooling stride size. If N
 * is 2, then is {pool_stride_Height, pool_stride_Width}.
 * @param padding_size Vector of N*2 ints {head_pad_d1, head_pad_d2, ...,
 * head_pad_dN, tail_pad_d1, tail_pad_d2, ..., tail_pad_dN}. If N is 2, then is
 * {pad_height_top, pad_width_left, pad_height_bottom, pad_width_right]}.
 * @param pool_type The type of pooling operator, currently support "max" and
 * "avg".
 * @param axis Vector of axes of the tensor for pooling.
 * @param ceil_mode Whether to use ceil when calculating the output size.
 * @param exclusive Whether include padding in the calculation'.
 * @param output_name the name of the output tensor after padding and pooling.
 *
 * @return the vector of padding tensor and pooling tensor
 */
std::vector<Tensor> PoolImpl(const Tensor &tensor,
                             const std::vector<int> &kernel_size,
                             const std::vector<int> &stride_size,
                             const std::vector<int> &padding_size,
                             const std::string &pooling_type,
                             const std::vector<int> &axis,
                             bool ceil_mode,
                             bool exclusive,
                             bool adaptive,
                             const std::string &output_name) {
  CHECK(!kernel_size.empty()) << "Pooling kernel_size should not be empty\n";
  int k_size = kernel_size.size();
  int x_size = tensor->shape.size();
  CHECK_EQ(stride_size.size(), k_size)
      << "Pooling stride_size must have same elements as kernel\n";
  CHECK_EQ(padding_size.size(), k_size * 2)
      << "Pooling padding_size must have double elements as kernel\n";
  CHECK_EQ(axis.size(), k_size) << "Axis must have same elements as kernel\n";

  std::string pool_type;
  std::transform(pooling_type.begin(),
                 pooling_type.end(),
                 std::back_inserter(pool_type),
                 [](unsigned char c) { return std::tolower(c); });
  CHECK(pool_type == "max" || pool_type == "avg")
      << "pool_type for pool2d should be max or avg.\n";

  std::vector<Var> daxis;
  std::vector<Expr> kernel(k_size);
  std::vector<Expr> stride(k_size);
  std::vector<Expr> pad_head(k_size);
  std::vector<Expr> pad_tail(k_size);
  std::vector<Expr> pad_before(x_size, Expr(0));
  std::vector<Expr> pad_after(x_size, Expr(0));
  std::vector<Expr> out_shape = tensor->shape;

  bool do_pad = false;
  for (int i = 0; i < k_size; i++) {
    int ii = axis[i];
    kernel[i] = Expr(kernel_size[i]);
    stride[i] = Expr(stride_size[i]);
    pad_head[i] = Expr(padding_size[i]);
    pad_tail[i] = Expr(padding_size[i + k_size]);
    do_pad = (do_pad) ? do_pad : (padding_size[i] || padding_size[i + k_size]);

    if (ceil_mode) {
      pad_tail[i] = common::AutoSimplify(pad_tail[i] + stride[i] - 1);
    }

    daxis.emplace_back(Var(kernel[i], UniqName("kernel_idx")));

    pad_before[ii] = pad_head[i];
    pad_after[ii] = pad_tail[i];

    auto out_dim = common::AutoSimplify(
        (tensor->shape[ii] - kernel[i] + pad_head[i] + pad_tail[i]) /
            stride[i] +
        1);

    out_shape[ii] = out_dim;
  }

  do_pad = do_pad || (ceil_mode && stride_size[0] > 1);
  Tensor temp;
  Tensor res;
  if (pool_type == "max") {
    Expr min_value = lang::min_value(tensor->type());
    // Pad the input tensor with the pad_value of type's minimum value
    temp = do_pad ? Pad(tensor,
                        pad_before,
                        pad_after,
                        min_value,
                        UniqName("pad_temp"))
                  : tensor;
    res = Compute(
        out_shape,
        [=](const std::vector<Expr> &output) {
          std::vector<Expr> indices;
          for (auto &var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii = axis[i];
            indices[ii] = output[ii] * stride[i] + daxis[i];
          }

          return lang::ReduceMax(temp(indices), {daxis}, min_value);
        },
        output_name);
  } else if (pool_type == "avg") {
    // Pad the input tensor with pad_value zero
    temp = do_pad ? Pad(tensor, pad_before, pad_after, 0, UniqName("pad_temp"))
                  : tensor;
    res = Compute(
        out_shape,
        [=](const std::vector<Expr> &output) {
          std::vector<Expr> indices;
          for (const Expr &var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii = axis[i];
            indices[ii] = output[ii] * stride[i] + daxis[i];
          }

          if (exclusive) {
            std::vector<Expr> start(k_size);
            std::vector<Expr> end(k_size);
            auto temp_factor = make_const(Int(32), 1);
            for (int i = 0; i < k_size; i++) {
              int ii = axis[i];
              start[i] =
                  common::AutoSimplify(output[ii] * stride[i] - pad_head[i]);
              end[i] = Min::Make(start[i] + kernel[i], tensor->shape[ii]);
              start[i] = Max::Make(start[i], make_const(Int(32), 0));
              temp_factor = temp_factor * (end[i] - start[i]);
            }
            common::AutoSimplify(temp_factor);
            Expr divide_factor = Max::Make(temp_factor, make_const(Int(32), 1));
            return lang::ReduceSum(
                ir::Div::Make(temp(indices),
                              ir::Cast::Make(temp->type(), divide_factor)),
                {daxis});
          } else {
            auto temp_factor = make_const(Int(32), 1);
            for (int i = 0; i < k_size; i++) {
              temp_factor = temp_factor * kernel[i];
            }
            common::AutoSimplify(temp_factor);
            return lang::ReduceSum(
                ir::Div::Make(temp(indices),
                              ir::Cast::Make(temp->type(), temp_factor)),
                daxis);
          }
        },
        output_name);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
  }
  if (adaptive) {
    CHECK_EQ(pool_type, "avg");
    std::vector<Expr> out_shape = tensor->shape;
    CHECK_EQ(k_size, 2);
    CHECK_EQ(k_size, (int)axis.size());
    for (int i = 0; i < k_size; i++) {
      out_shape[axis[i]] = Expr(kernel_size[i]);
    }
    VLOG(4) << "PoolImpl out_shape: " << cinn::utils::Join(out_shape, ",");
    CHECK(!do_pad);
    temp = do_pad ? Pad(tensor, pad_before, pad_after, 0, UniqName("pad_temp"))
                  : tensor;
    std::vector<Var> reduce_axis;

    for (int i = 0; i < k_size; i++) {
      reduce_axis.emplace_back(
          Var(Expr(static_cast<int>(tensor->shape[axis[i]].get_constant()) /
                   kernel_size[i]),
              UniqName("adaptive_reduce")));
    }

    res = Compute(
        out_shape,
        [=](const std::vector<Expr> &output) {
          std::vector<Expr> indices;
          for (const Expr &var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            indices[axis[i]] =
                output[axis[i]] *
                    Expr(static_cast<int>(
                             tensor->shape[axis[i]].get_constant()) /
                         kernel_size[i]) +
                reduce_axis[i];
          }

          auto temp_factor = make_const(Int(32), 1);
          for (int i = 0; i < k_size; i++) {
            temp_factor =
                temp_factor *
                Expr(static_cast<int>(tensor->shape[axis[i]].get_constant()) /
                     kernel_size[i]);
          }
          common::AutoSimplify(temp_factor);
          Expr divide_factor = Max::Make(temp_factor, make_const(Int(32), 1));
          return lang::ReduceSum(
              ir::Div::Make(temp(indices),
                            ir::Cast::Make(temp->type(), divide_factor)),
              {reduce_axis});
        },
        output_name);
  }
  if (do_pad) {
    return {res, temp};
  } else {
    return {res};
  }
}

std::vector<Tensor> Pool1d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           const std::string &output_name) {
  int width_axis = -1;
  if (data_format == "NCW") {
    width_axis = 2;
  } else if (data_format == "NWC") {
    width_axis = 1;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK_EQ(tensor->shape.size(), 3U)
      << "pool1d requires tensor's shape_size to be 3\n";
  std::vector<int> axis = {width_axis};
  return PoolImpl(tensor,
                  kernel_size,
                  stride_size,
                  padding_size,
                  pool_type,
                  axis,
                  ceil_mode,
                  exclusive,
                  false,
                  output_name);
}

std::vector<Tensor> GlobalPool2d(const Tensor &tensor,
                                 const std::string &pool_type,
                                 const std::string &output_name) {
  // TODO(hp03): 1. check warp shuffle is supported!
  // TODO(hp03): 2. using `cub` with NVRTC
  Expr extend = tensor->shape[2] * tensor->shape[3];
  if (pool_type == "max") {
    auto temp = Compute(
        {tensor->shape[0], tensor->shape[1], Expr(32)},
        [=](Expr n, Expr c, Expr k) -> Expr {
          Expr offset = common::IndiceToAbsOffset(tensor->shape,
                                                  {n, c, Expr(0), Expr(0)});
          return lang::CallExtern(
              "cinn_warp_reduce_max_" + Type2StrForNN(tensor->type()),
              {tensor, offset, extend});
        },
        UniqName(output_name + "_temp"));
    temp->WithBuffer(tensor->type());
    auto ret = Compute(
        {tensor->shape[0], tensor->shape[1]},
        [=](Expr n, Expr c) -> Expr {
          return temp({n, c, Expr(0)});
        },
        UniqName(output_name));
    return {ret, temp};
  } else if (pool_type == "avg") {
    auto temp = Compute(
        {tensor->shape[0], tensor->shape[1], Expr(32)},
        [=](Expr n, Expr c, Expr k) -> Expr {
          Expr offset = common::IndiceToAbsOffset(tensor->shape,
                                                  {n, c, Expr(0), Expr(0)});
          return lang::CallExtern(
              "cinn_warp_reduce_avg_" + Type2StrForNN(tensor->type()),
              {tensor, offset, extend});
        },
        UniqName(output_name + "_temp"));
    temp->WithBuffer(tensor->type());
    auto ret = Compute(
        {tensor->shape[0], tensor->shape[1]},
        [=](Expr n, Expr c) -> Expr {
          return temp({n, c, Expr(0)});
        },
        UniqName(output_name));
    return {ret, temp};
  } else {
    LOG(FATAL) << "unsupported pooling type.";
  }
  return {};
}

std::vector<Tensor> Pool2d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           bool adaptive,
                           const std::string &output_name) {
  int height_axis = -1;
  int width_axis = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis = 2;
  } else if (data_format == "AnyLayout") {
    height_axis = 2;
    width_axis = 3;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK(tensor->shape.size() == 4U || tensor->shape.size() == 5U)
      << "pool2d requires tensor's shape_size to be 4 or 5\n";
  std::vector<int> axis = {height_axis, width_axis};
  return PoolImpl(tensor,
                  kernel_size,
                  stride_size,
                  padding_size,
                  pool_type,
                  axis,
                  ceil_mode,
                  exclusive,
                  adaptive,
                  output_name);
}

std::vector<Tensor> Pool3d(const Tensor &tensor,
                           const std::vector<int> &kernel_size,
                           const std::vector<int> &stride_size,
                           const std::vector<int> &padding_size,
                           const std::string &pool_type,
                           bool ceil_mode,
                           bool exclusive,
                           const std::string &data_format,
                           const std::string &output_name) {
  int height_axis = -1;
  int width_axis = -1;
  int depth_axis = -1;
  if (data_format == "NCDHW") {
    depth_axis = 2;
    height_axis = 3;
    width_axis = 4;
  } else if (data_format == "NDHWC") {
    depth_axis = 1;
    height_axis = 2;
    width_axis = 3;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  CHECK_EQ(tensor->shape.size(), 5U)
      << "pool1d requires tensor's shape_size to be 5\n";
  std::vector<int> axis = {depth_axis, height_axis, width_axis};
  return PoolImpl(tensor,
                  kernel_size,
                  stride_size,
                  padding_size,
                  pool_type,
                  axis,
                  ceil_mode,
                  exclusive,
                  false,
                  UniqName(output_name));
}

Tensor DropoutInfer(const ir::Tensor &tensor,
                    float dropout_prob,
                    const std::string &dropout_implementation,
                    const std::string &output_name) {
  if (dropout_implementation == "downgrade_in_infer") {
    return Compute(
        tensor->shape,
        [=](const std::vector<Expr> &indice) {
          return tensor(indice) *
                 common::make_const(tensor->type(), 1 - dropout_prob);
        },
        output_name);
  } else if (dropout_implementation == "upscale_in_train") {
    // The name here must be consistent, otherwise it cannot participate in the
    // fusion schedule.
    return Identity(tensor, output_name).front();
  } else {
    LOG(FATAL) << "dropout_implementation attr must be 'downgrade_in_infer' or "
                  "'upscale_in_train'\n";
  }
}

ir::Tensor Select(const ir::Tensor &condition,
                  const ir::Tensor &true_value,
                  const ir::Tensor &false_value,
                  const std::string &output_name) {
  CHECK(condition->type().is_bool())
      << "The condition tensor type should be bool!";
  CHECK(condition->shape == true_value->shape &&
        true_value->shape == false_value->shape)
      << "The input tensor shape is not equal!";
  return lang::Compute(
      condition->shape,
      [=](const std::vector<Expr> &indice) {
        return common::select(
            condition(indice), true_value(indice), false_value(indice));
      },
      output_name);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
