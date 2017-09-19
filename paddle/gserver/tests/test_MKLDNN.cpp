/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "MKLDNNTester.h"
#include "ModelConfig.pb.h"
#include "paddle/math/MathUtils.h"

using namespace paddle;  // NOLINT

DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(use_gpu);
DECLARE_bool(use_mkldnn);

struct testFCDesc {
  int bs;
  int ic;
  int oc;
  int ih, iw;  // oh == ow == 1
};

void testFcLayer(const testFCDesc& pm) {
  const std::string compareTypes[] = {"mkldnn_fc", "fc"};
  TestConfig cfg;
  cfg.layerConfig.set_type(compareTypes[0]);
  cfg.layerConfig.set_size(pm.oc);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       /* size of weight= */ size_t(pm.oc * pm.ic * pm.ih * pm.iw)});
  cfg.layerConfig.add_inputs();

  MKLDNNTester tester;
  for (auto biasSize : {pm.oc, 0}) {
    cfg.biasSize = biasSize;
    TestConfig ref = cfg;
    ref.layerConfig.set_type(compareTypes[1]);
    for (auto bs : {pm.bs, 1}) {
      tester.run(cfg, ref, bs, pm.ih, pm.iw);
    }
  }
}

TEST(MKLDNNLayer, FcLayer) {
  testFcLayer({/*bs*/ 2, /*ic*/ 2, /*oc*/ 3, /*ih*/ 1, /*iw*/ 1});
  testFcLayer({/*bs*/ 3, /*ic*/ 7, /*oc*/ 19, /*ih*/ 1, /*iw*/ 1});
  testFcLayer({/*bs*/ 8, /*ic*/ 16, /*oc*/ 32, /*ih*/ 13, /*iw*/ 13});
  testFcLayer({/*bs*/ 4, /*ic*/ 12, /*oc*/ 18, /*ih*/ 13, /*iw*/ 11});
  testFcLayer({/*bs*/ 2, /*ic*/ 64, /*oc*/ 32, /*ih*/ 16, /*iw*/ 16});
  testFcLayer({/*bs*/ 15, /*ic*/ 3, /*oc*/ 6, /*ih*/ 16, /*iw*/ 16});
}

struct testConvDesc {
  int bs, gp;
  int ic, ih, iw;
  int oc, oh, ow;
  int fh, fw;
  int ph, pw;
  int sh, sw;
  int dh, dw;
};

void testConvLayer(const testConvDesc& pm) {
  const std::string compareTypes[] = {"mkldnn_conv", "exconv"};
  TestConfig cfg;
  cfg.layerConfig.set_type(compareTypes[0]);
  cfg.layerConfig.set_num_filters(pm.oc);
  cfg.layerConfig.set_size(pm.oc * pm.oh * pm.ow);
  // cfg.layerConfig.set_partial_sum(1); // TODO: check it
  cfg.layerConfig.set_shared_biases(true);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       /* size of weight= */ size_t(pm.oc * pm.ic * pm.fh * pm.fw / pm.gp)});
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_groups(pm.gp);
  conv->set_img_size(pm.iw);
  conv->set_img_size_y(pm.ih);
  conv->set_output_x(pm.ow);
  conv->set_output_y(pm.oh);
  conv->set_filter_size(pm.fw);
  conv->set_filter_size_y(pm.fh);
  conv->set_channels(pm.ic);
  conv->set_padding(pm.pw);
  conv->set_padding_y(pm.ph);
  conv->set_stride(pm.sw);
  conv->set_stride_y(pm.sh);
  conv->set_dilation(pm.dw);
  conv->set_dilation_y(pm.dh);
  conv->set_caffe_mode(true);
  conv->set_filter_channels(conv->channels() / conv->groups());
  CHECK_EQ(conv->filter_channels() * pm.gp, conv->channels())
      << "it is indivisible";

  int fh = (pm.fh - 1) * pm.dh + 1;
  int fw = (pm.fw - 1) * pm.dw + 1;
  int ow = outputSize(pm.iw, fw, pm.pw, pm.sw, true);
  int oh = outputSize(pm.ih, fh, pm.ph, pm.sh, true);
  CHECK_EQ(ow, pm.ow) << "output size check failed";
  CHECK_EQ(oh, pm.oh) << "output size check failed";

  MKLDNNTester tester;
  for (auto biasSize : {pm.oc, 0}) {
    cfg.biasSize = biasSize;
    TestConfig ref = cfg;
    ref.layerConfig.set_type(compareTypes[1]);
    for (auto bs : {pm.bs, 1}) {
      tester.run(cfg, ref, bs, pm.ih, pm.iw);
    }
  }
}

TEST(MKLDNNLayer, ConvLayer) {
  /* bs, gp, ic, ih, iw, oc, oh, ow, fh, fw, ph, pw, sh, sw, dh, dw */
  testConvLayer({2, 1, 3, 32, 32, 16, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({2, 1, 8, 16, 16, 8, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({3, 1, 16, 32, 32, 3, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({8, 1, 16, 18, 18, 32, 18, 18, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({16, 1, 1, 42, 31, 32, 23, 11, 4, 5, 3, 2, 2, 3, 1, 1});
  testConvLayer({2, 1, 8, 16, 16, 8, 8, 8, 3, 3, 1, 1, 2, 2, 1, 1});
  testConvLayer({3, 1, 8, 13, 13, 8, 7, 7, 3, 3, 1, 1, 2, 2, 1, 1});
  // with groups
  testConvLayer({2, 2, 4, 5, 5, 8, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({2, 3, 3, 5, 5, 3, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1});
  testConvLayer({4, 4, 16, 3, 3, 16, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1});
}

struct testPoolDesc {
  int bs, ch;  // input channel and output channel are the same
  int ih, iw;
  int oh, ow;
  int fh, fw;
  int ph, pw;
  int sh, sw;
};

void testPoolLayer(const testPoolDesc& pm) {
  const std::string compareTypes[] = {"mkldnn_pool", "pool"};
  TestConfig cfg;
  cfg.layerConfig.set_type(compareTypes[0]);
  cfg.layerConfig.set_size(pm.ch * pm.oh * pm.ow);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ch * pm.ih * pm.iw),
       0});
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();
  // pool->set_pool_type(poolType);
  pool->set_channels(pm.ch);
  pool->set_img_size(pm.iw);
  pool->set_img_size_y(pm.ih);
  pool->set_output_x(pm.ow);
  pool->set_output_y(pm.oh);
  pool->set_size_x(pm.fw);
  pool->set_size_y(pm.fh);
  pool->set_padding(pm.pw);
  pool->set_padding_y(pm.ph);
  pool->set_stride(pm.sw);
  pool->set_stride_y(pm.sh);

  int oh = outputSize(pm.ih, pm.fh, pm.ph, pm.sh, false);
  int ow = outputSize(pm.iw, pm.fw, pm.pw, pm.sw, false);
  CHECK_EQ(ow, pm.ow) << "output size check failed";
  CHECK_EQ(oh, pm.oh) << "output size check failed";

  MKLDNNTester tester;
  for (auto type : {"max-projection", "avg-projection"}) {
    pool->set_pool_type(type);
    TestConfig ref = cfg;
    ref.layerConfig.set_type(compareTypes[1]);
    for (auto bs : {pm.bs, 1}) {
      tester.run(cfg, ref, bs, pm.ih, pm.iw);
    }
  }
}

TEST(MkldnnLayer, PoolLayer) {
  /* bs, ch, ih, iw, oh, ow, fh, fw, ph, pw, sh, sw*/
  testPoolLayer({2, 1, 4, 4, 2, 2, 3, 3, 0, 0, 2, 2});
  testPoolLayer({10, 8, 16, 16, 8, 8, 2, 2, 0, 0, 2, 2});
  testPoolLayer({4, 2, 5, 5, 3, 3, 3, 3, 1, 1, 2, 2});
  testPoolLayer({8, 16, 56, 56, 28, 28, 3, 3, 0, 0, 2, 2});
  testPoolLayer({8, 16, 14, 14, 7, 7, 3, 3, 0, 0, 2, 2});
  testPoolLayer({4, 16, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1});
  testPoolLayer({4, 2, 5, 5, 3, 3, 5, 5, 1, 1, 1, 1});
  testPoolLayer({2, 8, 56, 56, 29, 29, 3, 3, 1, 1, 2, 2});
}

// TODO(TJ): add branch test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  FLAGS_use_gpu = false;
  FLAGS_use_mkldnn = true;
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
