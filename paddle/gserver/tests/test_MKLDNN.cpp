/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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
#include <paddle/utils/PythonUtil.h>
#include <string>
#include <vector>
#include "MKLDNNTester.h"
#include "ModelConfig.pb.h"
#include "paddle/gserver/activations/MKLDNNActivation.h"
#include "paddle/math/MathUtils.h"

using namespace paddle;  // NOLINT

DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(use_gpu);
DECLARE_bool(use_mkldnn);

#define RUN_MKLDNN_TEST(DNN_CONFIG, REF_CONFIG, DESC)         \
  MKLDNNTester tester;                                        \
  for (auto bs : {DESC.bs, 1}) {                              \
    tester.run(DNN_CONFIG, REF_CONFIG, bs, DESC.ih, DESC.iw); \
  }

#define RUN_MKLDNN_TEST_LAYER(DNN_CONFIG, REF_TYPE, DESC) \
  TestConfig ref = DNN_CONFIG;                            \
  ref.layerConfig.set_type(REF_TYPE);                     \
  RUN_MKLDNN_TEST(DNN_CONFIG, ref, DESC)

struct testFcDesc {
  int bs;
  int ic;
  int ih, iw;  // oh == ow == 1
  int oc;
};

static void getMKLDNNFcConfig(TestConfig& cfg, const testFcDesc& pm) {
  cfg.layerConfig.set_type("mkldnn_fc");
  cfg.layerConfig.set_active_type("relu");
  cfg.layerConfig.set_size(pm.oc);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       /* size of weight= */ size_t(pm.oc * pm.ic * pm.ih * pm.iw)});
  cfg.layerConfig.add_inputs();
}

void testFcLayer(const testFcDesc& pm) {
  TestConfig dnnConfig;
  getMKLDNNFcConfig(dnnConfig, pm);
  for (auto biasSize : {pm.oc, 0}) {
    dnnConfig.biasSize = biasSize;
    RUN_MKLDNN_TEST_LAYER(dnnConfig, "fc", pm)
  }
}

TEST(MKLDNNLayer, FcLayer) {
  /* bs, ic, ih, iw, oc */
  testFcLayer({2, 2, 1, 1, 3});
  testFcLayer({3, 7, 1, 1, 19});
  testFcLayer({8, 16, 13, 13, 32});
  testFcLayer({4, 12, 13, 13, 18});
  testFcLayer({2, 64, 16, 16, 32});
  testFcLayer({15, 3, 16, 16, 6});
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

static void getMKLDNNConvConfig(TestConfig& cfg, const testConvDesc& pm) {
  cfg.layerConfig.set_type("mkldnn_conv");
  cfg.layerConfig.set_active_type("relu");
  cfg.layerConfig.set_num_filters(pm.oc);
  cfg.layerConfig.set_size(pm.oc * pm.oh * pm.ow);
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
}

void testConvLayer(const testConvDesc& pm) {
  TestConfig dnnConfig;
  getMKLDNNConvConfig(dnnConfig, pm);
  for (auto biasSize : {pm.oc, 0}) {
    dnnConfig.biasSize = biasSize;
    RUN_MKLDNN_TEST_LAYER(dnnConfig, "exconv", pm)
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
  int bs, ic;  // input channel and output channel are the same
  int ih, iw;
  int oh, ow;
  int fh, fw;
  int ph, pw;
  int sh, sw;
};

static void getMKLDNNPoolConfig(TestConfig& cfg, const testPoolDesc& pm) {
  cfg.layerConfig.set_type("mkldnn_pool");
  cfg.layerConfig.set_active_type("relu");
  cfg.layerConfig.set_size(pm.ic * pm.oh * pm.ow);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       0});
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();
  pool->set_pool_type("avg-projection");
  pool->set_channels(pm.ic);
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
}

void testPoolLayer(const testPoolDesc& pm) {
  TestConfig dnnConfig;
  getMKLDNNPoolConfig(dnnConfig, pm);
  LayerInputConfig* input = dnnConfig.layerConfig.mutable_inputs(0);
  PoolConfig* pool = input->mutable_pool_conf();
  for (auto type : {"max-projection", "avg-projection"}) {
    pool->set_pool_type(type);
    RUN_MKLDNN_TEST_LAYER(dnnConfig, "pool", pm)
  }
}

TEST(MKLDNNLayer, PoolLayer) {
  /* bs, ch, ih, iw, oh, ow, fh, fw, ph, pw, sh, sw */
  testPoolLayer({2, 1, 4, 4, 2, 2, 3, 3, 0, 0, 2, 2});
  testPoolLayer({10, 8, 16, 16, 8, 8, 2, 2, 0, 0, 2, 2});
  testPoolLayer({4, 2, 5, 5, 3, 3, 3, 3, 1, 1, 2, 2});
  testPoolLayer({8, 16, 56, 56, 28, 28, 3, 3, 0, 0, 2, 2});
  testPoolLayer({8, 16, 14, 14, 7, 7, 3, 3, 0, 0, 2, 2});
  testPoolLayer({4, 16, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1});
  testPoolLayer({4, 2, 5, 5, 3, 3, 5, 5, 1, 1, 1, 1});
  testPoolLayer({2, 8, 56, 56, 29, 29, 3, 3, 1, 1, 2, 2});
}

struct testBatchNormDesc {
  int bs;
  int ic;
  int ih, iw;
};

static void getMKLDNNBatchNormConfig(TestConfig& cfg,
                                     const testBatchNormDesc& pm) {
  cfg.layerConfig.set_size(pm.ic * pm.ih * pm.iw);
  cfg.layerConfig.set_type("mkldnn_batch_norm");
  cfg.biasSize = pm.ic;
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       /* size of weight= */ size_t(pm.ic)});
  cfg.inputDefs.push_back(
      {INPUT_DATA, "layer_1_moving_mean", 1, size_t(pm.ic)});
  cfg.inputDefs.back().isStatic = true;
  cfg.inputDefs.push_back({INPUT_DATA, "layer_2_moving_var", 1, size_t(pm.ic)});
  cfg.inputDefs.back().isStatic = true;
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  cfg.layerConfig.set_active_type("relu");
  cfg.layerConfig.add_inputs();
  cfg.layerConfig.add_inputs();
  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(pm.ic);
  img_conf->set_img_size_y(pm.ih);
  img_conf->set_img_size(pm.iw);
}

void testBatchNormLayer(const testBatchNormDesc& pm) {
  TestConfig dnnConfig;
  getMKLDNNBatchNormConfig(dnnConfig, pm);
  TestConfig refConfig = dnnConfig;
  refConfig.layerConfig.set_type("batch_norm");
  // for PASS_TRAIN, use_global_stats always should be false, and batchsize != 1
  VLOG(MKLDNN_TESTS) << "check train phase";
  dnnConfig.layerConfig.set_use_global_stats(false);
  refConfig.layerConfig.set_use_global_stats(false);
  MKLDNNTester tester;
  tester.run(dnnConfig, refConfig, pm.bs, pm.ih, pm.iw, PASS_TRAIN);
  // for PASS_TEST, check use_global_stats true and false, and batchsize 1
  VLOG(MKLDNN_TESTS) << "check test phase";
  for (auto useGS : {false, true}) {
    dnnConfig.layerConfig.set_use_global_stats(useGS);
    refConfig.layerConfig.set_use_global_stats(useGS);
    MKLDNNTester tester;
    for (auto bs : {pm.bs, 1}) {
      tester.run(dnnConfig, refConfig, bs, pm.ih, pm.iw, PASS_TEST);
    }
  }
}

TEST(MKLDNNLayer, BatchNormLayer) {
  testBatchNormLayer({4, 10, 6, 6});
  testBatchNormLayer({16, 32, 16, 16});
  testBatchNormLayer({4, 16, 8, 10});
}

struct testLRNDesc {
  int bs, ic, ih, iw;
  float scale, pow;
  int localSize;
};

void getMKLDNNLRNConfig(TestConfig& cfg, const testLRNDesc& pm) {
  cfg.layerConfig.set_type("mkldnn_lrn");
  cfg.layerConfig.set_active_type("relu");
  size_t layerSize = pm.ic * pm.ih * pm.iw;
  cfg.inputDefs.push_back({INPUT_DATA, "layer_0", layerSize, 0});
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  NormConfig* norm = input->mutable_norm_conf();
  norm->set_channels(pm.ic);
  norm->set_size(pm.localSize);
  norm->set_scale(pm.scale);
  norm->set_pow(pm.pow);
  norm->set_blocked(0);
  norm->set_img_size(pm.iw);
  norm->set_img_size_y(pm.ih);
  norm->set_output_x(norm->img_size());
  norm->set_output_y(norm->img_size_y());
  cfg.layerConfig.set_size(layerSize);
  cfg.biasSize = 0;
}

void testLRNLayer(const testLRNDesc& pm) {
  TestConfig dnnConfig;
  getMKLDNNLRNConfig(dnnConfig, pm);
  // mkldnn_lrn <==> norm with cmrnorm-projection type
  TestConfig refConfig = dnnConfig;
  refConfig.layerConfig.set_type("norm");
  LayerInputConfig* input = refConfig.layerConfig.mutable_inputs(0);
  NormConfig* norm = input->mutable_norm_conf();
  norm->set_norm_type("cmrnorm-projection");
  norm->set_scale(norm->scale() / norm->size());
  RUN_MKLDNN_TEST(dnnConfig, refConfig, pm)
}

TEST(MKLDNNLayer, LRNLayer) {
  testLRNLayer({4, 10, 12, 12, 0.001f, 0.75f, 5});
  testLRNLayer({2, 32, 6, 6, 0.001f, 0.75f, 5});
  testLRNLayer({4, 16, 8, 10, 0.01f, 0.5f, 5});
}

struct testImageDesc {
  int bs, ic, ih, iw;
};

static void getAddtoConfig(TestConfig& cfg,
                           const testImageDesc& pm,
                           const size_t nInputs = 1) {
  cfg.biasSize = 0;
  cfg.layerConfig.set_type("addto");
  size_t layerSize = pm.ic * pm.ih * pm.iw;
  cfg.layerConfig.set_size(layerSize);
  cfg.layerConfig.set_active_type("relu");
  for (size_t i = 0; i < nInputs; ++i) {
    std::stringstream ss;
    ss << "layer_" << i;
    cfg.inputDefs.push_back({INPUT_DATA, ss.str(), layerSize, 0});
    LayerInputConfig* input = cfg.layerConfig.add_inputs();
    ImageConfig* img_conf = input->mutable_image_conf();
    img_conf->set_channels(pm.ic);
    img_conf->set_img_size_y(pm.ih);
    img_conf->set_img_size(pm.iw);
  }
}

void testAddtoLayer(const testImageDesc& pm, const size_t nInputs) {
  CHECK_GE(nInputs, 1UL);
  TestConfig dnnConfig;
  getAddtoConfig(dnnConfig, pm, nInputs);
  dnnConfig.layerConfig.set_type("mkldnn_addto");
  for (auto withBias : {false, true}) {
    dnnConfig.biasSize = withBias ? pm.ic * pm.ih * pm.iw : 0;
    RUN_MKLDNN_TEST_LAYER(dnnConfig, "addto", pm)
  }
}

TEST(MKLDNNLayer, AddtoLayer) {
  testAddtoLayer({16, 5, 14, 14}, 1);
  testAddtoLayer({8, 10, 8, 8}, 2);
  testAddtoLayer({4, 12, 1, 1}, 3);
}

static void getMKLDNNConcatConfig(TestConfig& cfg,
                                  const std::vector<testImageDesc>& inputs) {
  CHECK_GE(inputs.size(), 2UL) << "at least two inputs";
  int oc = inputs[0].ic;
  for (size_t i = 1; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i].bs, inputs[0].bs);
    CHECK_EQ(inputs[i].ih, inputs[0].ih);
    CHECK_EQ(inputs[i].iw, inputs[0].iw);
    oc += inputs[i].ic;
  }
  cfg.biasSize = 0;
  cfg.layerConfig.set_type("mkldnn_concat");
  cfg.layerConfig.set_size(oc * inputs[0].ih * inputs[0].iw);
  cfg.layerConfig.set_active_type("relu");
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::stringstream ss;
    ss << "layer_" << i;
    cfg.inputDefs.push_back(
        {INPUT_DATA,
         ss.str(),
         (size_t)(inputs[i].ic) * inputs[i].ih * inputs[i].iw,
         0});
    LayerInputConfig* input = cfg.layerConfig.add_inputs();
    ImageConfig* img_conf = input->mutable_image_conf();
    img_conf->set_channels(inputs[i].ic);
    img_conf->set_img_size_y(inputs[i].ih);
    img_conf->set_img_size(inputs[i].iw);
  }
}

void testConcatLayer(const std::vector<testImageDesc>& inputs) {
  TestConfig dnnConfig;
  getMKLDNNConcatConfig(dnnConfig, inputs);
  RUN_MKLDNN_TEST_LAYER(dnnConfig, "concat", inputs[0])
}

TEST(MKLDNNLayer, ConcatLayer) {
  testConcatLayer({{64, 128, 1, 1}, {64, 32, 1, 1}, {64, 64, 1, 1}});
  testConcatLayer({{32, 100, 8, 8}, {32, 10, 8, 8}});
}

void testActivation(std::string actType, const testImageDesc& pm) {
  // TODO(TJ): remove me when paddle support elu activation
  if (actType == "mkldnn_elu") {
    return;
  }
  const std::string compareTypes[] = {actType, actType.erase(0, 7)};
  TestConfig cfg;
  getAddtoConfig(cfg, pm);
  TestConfig ref = cfg;
  cfg.layerConfig.set_active_type(compareTypes[0]);
  ref.layerConfig.set_active_type(compareTypes[1]);
  RUN_MKLDNN_TEST(cfg, ref, pm)
}

TEST(MKLDNNActivation, Activations) {
  auto types = MKLDNNActivation::getAllRegisteredTypes();
  for (auto type : types) {
    /* bs, c, h, w*/
    testActivation(type, {16, 64, 32, 32});
    testActivation(type, {2, 8, 1, 1});
  }
}

DECLARE_string(config_args);
TEST(MKLDNNNet, net) {
  std::vector<std::string> cases = {"simple", "branch"};
  for (auto name : cases) {
    std::string config = "./gserver/tests/mkldnn_" + name + "_net.conf";
    for (auto channels : {2, 32}) {
      std::ostringstream oss;
      oss << "channels=" << channels;
      FLAGS_config_args = oss.str();
      MKLDNNTester::runNetTest(config);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  FLAGS_use_gpu = false;
  FLAGS_use_mkldnn = true;
  initMain(argc, argv);
  initPython(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
