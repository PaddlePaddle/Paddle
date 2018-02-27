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

#ifdef PADDLE_WITH_CUDA
#include <cudnn.h>
#endif
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/math/MathUtils.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_double(checkgrad_eps);
DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(prev_batch_state);

TEST(Operator, dot_mul) {
  TestConfig config;
  config.layerConfig.set_size(10);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  OperatorConfig& operatorConf = *config.layerConfig.add_operator_confs();
  operatorConf.set_type("dot_mul");
  operatorConf.set_dotmul_scale(-1);

  testOperatorGrad(config, operatorConf, 100, false, false);
}

TEST(Projection, context) {
  for (auto contextStart : {-5, -3, -1, 0, 3}) {
    for (auto contextLength : {1, 2, 5, 7}) {
      for (auto batchSize : {1, 2, 5, 20}) {
        for (auto trainablePadding : {false, true}) {
          LOG(INFO) << " contextStart=" << contextStart
                    << " contextLength=" << contextLength
                    << " batchSize=" << batchSize
                    << " trainablePadding=" << trainablePadding;
          ProjectionConfig conf;
          conf.set_type("context");
          conf.set_input_size(10);
          conf.set_context_start(contextStart);
          conf.set_context_length(contextLength);
          conf.set_trainable_padding(trainablePadding);
          conf.set_output_size(conf.context_length() * conf.input_size());
          int pad =
              std::max(0, -conf.context_start()) +
              std::max(0, conf.context_start() + conf.context_length() - 1);
          for (auto useGpu : {false, true}) {
            testProjectionGrad(
                conf,
                INPUT_SEQUENCE_DATA,
                trainablePadding ? conf.input_size() * pad : 0,
                batchSize,
                useGpu,
                contextStart + contextLength <= 1);  // = testState
          }
        }
      }
    }
  }
}

TEST(Projection, trans_fc) {
  ProjectionConfig conf;
  conf.set_type("trans_fc");
  conf.set_input_size(50);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 1000,
                       /* batchSize */ 100,
                       useGpu);
  }
}

TEST(Projection, fc) {
  ProjectionConfig conf;
  conf.set_type("fc");
  conf.set_input_size(10);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 200,
                       /* batchSize */ 100,
                       useGpu);
  }
}

TEST(Projection, dot_mul) {
  ProjectionConfig conf;
  conf.set_type("dot_mul");
  conf.set_input_size(20);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 20,
                       /* batchSize */ 100,
                       useGpu);
  }
}

TEST(Projection, table) {
  ProjectionConfig conf;
  conf.set_type("table");
  conf.set_input_size(10);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_LABEL,
                       /* parameterSize */ 200,
                       /* batchSize */ 100,
                       useGpu);
  }
}

TEST(Projection, identity) {
  ProjectionConfig conf;
  conf.set_type("identity");
  conf.set_input_size(10);
  conf.set_output_size(10);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 0,
                       /* batchSize */ 100,
                       useGpu);
  }
}

TEST(Projection, slice) {
  ProjectionConfig conf;
  conf.set_type("slice");
  conf.set_input_size(100);
  SliceConfig& slice1 = *conf.add_slices();
  slice1.set_start(10);
  slice1.set_end(20);
  SliceConfig& slice2 = *conf.add_slices();
  slice2.set_start(50);
  slice2.set_end(70);
  conf.set_output_size(30);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 0,
                       /* batchSize */ 10,
                       useGpu);
  }
}

TEST(Projection, scaling) {
  ProjectionConfig conf;
  conf.set_type("scaling");
  conf.set_input_size(10);
  conf.set_output_size(10);
  for (auto useGpu : {false}) {
    testProjectionGrad(conf,
                       INPUT_DATA,
                       /* parameterSize */ 1,
                       /* batchSize */ 100,
                       useGpu);
  }
}

void testProjectionConv(size_t groups, bool isDeconv) {
  const int NUM_FILTERS = 18;
  const int FILTER_SIZE = 2;
  const int FILTER_SIZE_Y = 2;
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 16;

#if CUDNN_VERSION >= 6000
  const int DILATION = 2;
#else
  const int DILATION = 1;
#endif

  ProjectionConfig conf;
  if (isDeconv) {
    conf.set_type("convt");
  } else {
    conf.set_type("conv");
  }
  conf.set_num_filters(NUM_FILTERS);

  ConvConfig* conv = conf.mutable_conv_conf();
  conv->set_filter_size(FILTER_SIZE);
  conv->set_filter_size_y(FILTER_SIZE_Y);
  conv->set_channels(CHANNELS);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_dilation(DILATION);
  conv->set_dilation_y(DILATION);
  conv->set_groups(groups);
  if (isDeconv) {
    conv->set_filter_channels(NUM_FILTERS / conv->groups());
  } else {
    conv->set_filter_channels(conv->channels() / conv->groups());
  }
  conv->set_img_size(IMAGE_SIZE);
  int output_x = outputSize(conv->img_size(),
                            (conv->filter_size() - 1) * DILATION + 1,
                            conv->padding(),
                            conv->stride(),
                            /* caffeMode */ true);
  int output_y = outputSize(conv->img_size(),
                            (conv->filter_size_y() - 1) * DILATION + 1,
                            conv->padding_y(),
                            conv->stride_y(),
                            /* caffeMode */ true);
  conv->set_output_x(output_x);
  conv->set_output_y(output_y);
  LOG(INFO) << "DILATION:" << DILATION << "; output_x: " << output_x
            << "; output_y: " << output_y;
  if (isDeconv) {
    int deconv_image_x = imageSize(output_x,
                                   (conv->filter_size() - 1) * DILATION + 1,
                                   conv->padding(),
                                   conv->stride(),
                                   /* caffeMode */ true);
    int deconv_image_y = imageSize(output_y,
                                   (conv->filter_size_y() - 1) * DILATION + 1,
                                   conv->padding_y(),
                                   conv->stride_y(),
                                   /* caffeMode */ true);

    LOG(INFO) << " deconv_image_x: " << deconv_image_x
              << "; deconv_image_y: " << deconv_image_y;
    conf.set_input_size(output_x * output_y * CHANNELS);
    conf.set_output_size(deconv_image_x * deconv_image_y * NUM_FILTERS);
  } else {
    conf.set_input_size(IMAGE_SIZE * IMAGE_SIZE * CHANNELS);
    conf.set_output_size(output_x * output_y * NUM_FILTERS);
  }

  testProjectionGrad(conf,
                     INPUT_DATA,
                     /* parameterSize */ NUM_FILTERS * CHANNELS * FILTER_SIZE *
                         FILTER_SIZE_Y / groups,
                     /* batchSize */ 100,
                     true,
                     false,
                     NUM_FILTERS,
                     true);
}

#ifdef PADDLE_WITH_CUDA
TEST(Projection, conv) {
  /// test ConvProjection
  testProjectionConv(1, false);
  testProjectionConv(3, false);
  /// test ConvTransProjection
  testProjectionConv(1, true);
  testProjectionConv(3, true);
}
#endif

TEST(Layer, BilinearInterpLayer) {
  TestConfig config;
  config.layerConfig.set_type("bilinear_interp");
  config.biasSize = 0;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 4096, 0});

  LayerInputConfig* input = config.layerConfig.add_inputs();
  BilinearInterpConfig* bilinear = input->mutable_bilinear_interp_conf();
  ImageConfig* image = bilinear->mutable_image_conf();
  image->set_img_size(32);
  image->set_img_size_y(32);
  image->set_channels(4);

  for (auto useGpu : {false, true}) {
    for (auto outSize : {32, 64}) {
      bilinear->set_out_size_x(outSize);
      bilinear->set_out_size_y(outSize);
      testLayerGrad(config, "bilinear_interp", 10, false, useGpu);
    }
  }
}

TEST(Layer, concat) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("concat");
  config.layerConfig.set_size(15);
  config.layerConfig.set_active_type("sigmoid");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 5, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "concat", 100, false, useGpu);
  }
}

TEST(Layer, AddtoLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("addto");
  config.layerConfig.set_size(10);
  config.layerConfig.set_active_type("sigmoid");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "addto", 100, false, useGpu);
  }
}

TEST(Layer, CTCLayer) {
  TestConfig config;
  config.layerConfig.set_type("ctc");
  config.layerConfig.set_norm_by_times(false);
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_SEQUENCE_LABEL, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "ctc",
                  100,
                  /* trans */ false, /* useGpu */
                  useGpu);
  }
}

TEST(Layer, cosSimLayer) {
  TestConfig config;
  config.layerConfig.set_type("cos");
  config.layerConfig.set_size(1);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 50, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "cos", 100, false, useGpu);
  }
}

TEST(Layer, CosSimVecMatLayer) {
  TestConfig config;
  config.layerConfig.set_type("cos_vm");
  config.layerConfig.set_size(5);  // output size
  config.layerConfig.set_cos_scale(2.0);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 20, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 100, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "cos_vm", 100, false, useGpu);
  }
}

void testDepthwiseConvLayer(const string& type, bool useGpu) {
  TestConfig config;
  config.biasSize = 32;
  config.layerConfig.set_type(type);
  config.layerConfig.set_num_filters(32);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 2048, 192});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(3);
  conv->set_channels(16);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(16);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(16);
  conv->set_img_size_y(8);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /* caffeMode */ true));
  conv->set_output_y(outputSize(conv->img_size_y(),
                                conv->filter_size_y(),
                                conv->padding_y(),
                                conv->stride_y(),
                                /* caffeMode */ true));
  config.layerConfig.set_size(conv->output_x() * conv->output_y() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "depthwise_conv", 100, false, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "depthwise_conv", 2, false, useGpu, true, 0.02);
}

TEST(Layer, depthwiseConvLayer) {
  //  'depthwise_conv' is a sepecial case of 'exconv' whose
  //  groups size equals to the input channels size.
  testDepthwiseConvLayer("exconv", /* useGpu= */ false);
#ifdef PADDLE_WITH_CUDA
  testDepthwiseConvLayer("exconv", /* useGpu= */ true);
#endif
}

void testConvLayer(const string& type, bool trans, bool useGpu) {
  TestConfig config;
  config.biasSize = 16;
  config.layerConfig.set_type(type);
  config.layerConfig.set_num_filters(16);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  int dilation = 2;
  if (type == "cudnn_conv") {
#if CUDNN_VERSION >= 6000
    dilation = 2;
#else
    dilation = 1;
#endif
  }

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 768, 192});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(2);
  conv->set_channels(3);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_dilation(dilation);
  conv->set_dilation_y(dilation);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(16);
  conv->set_img_size_y(16);
  conv->set_output_x(outputSize(conv->img_size(),
                                (conv->filter_size() - 1) * dilation + 1,
                                conv->padding(),
                                conv->stride(),
                                /* caffeMode */ true));
  conv->set_output_y(outputSize(conv->img_size_y(),
                                (conv->filter_size_y() - 1) * dilation + 1,
                                conv->padding_y(),
                                conv->stride_y(),
                                /* caffeMode */ true));
  config.layerConfig.set_size(conv->output_x() * conv->output_y() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "conv", 100, trans, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "conv", 2, trans, useGpu, true, 0.02);
}

TEST(Layer, convLayer) {
  testConvLayer("exconv", /* trans= */ false, /* useGpu= */ false);
#ifdef PADDLE_WITH_CUDA
  testConvLayer("exconv", /* trans= */ false, /* useGpu= */ true);
  testConvLayer("cudnn_conv", /* trans= */ false, /* useGpu= */ true);
#endif
}

void testConvTransLayer(const string& type, bool trans, bool useGpu) {
  TestConfig config;
  config.biasSize = 3;
  config.layerConfig.set_type(type);
  config.layerConfig.set_num_filters(3);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1024, 384});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(4);
  conv->set_channels(16);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(3 / conv->groups());
  conv->set_img_size(16);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /* caffeMode */ true));

  config.layerConfig.set_size(conv->img_size() * conv->img_size() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "convTrans", 100, trans, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "convTrans", 2, trans, useGpu, true, 0.02);
}

TEST(Layer, convTransLayer) {
  for (auto useGpu : {false, true}) {
    testConvTransLayer("exconvt", /* trans= */ false, /* useGpu= */ useGpu);
  }
#ifdef PADDLE_WITH_CUDA
  testConvTransLayer("cudnn_convt", /* trans= */ false, /* useGpu= */ true);
#endif
}

TEST(Layer, blockExpandLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("blockexpand");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 6144, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  BlockExpandConfig* blockExpand = input->mutable_block_expand_conf();
  blockExpand->set_img_size_x(64);
  blockExpand->set_img_size_y(32);
  blockExpand->set_channels(3);
  blockExpand->set_padding_x(0);
  blockExpand->set_padding_y(0);
  blockExpand->set_block_x(4);
  blockExpand->set_block_y(32);
  blockExpand->set_stride_x(2);
  blockExpand->set_stride_y(2);
  blockExpand->set_output_x(outputSize(blockExpand->img_size_x(),
                                       blockExpand->block_x(),
                                       blockExpand->padding_x(),
                                       blockExpand->stride_x(),
                                       /* caffeMode */ false));
  blockExpand->set_output_y(outputSize(blockExpand->img_size_y(),
                                       blockExpand->block_y(),
                                       blockExpand->padding_y(),
                                       blockExpand->stride_y(),
                                       /* caffeMode */ false));
  config.layerConfig.set_size(blockExpand->block_x() * blockExpand->block_y() *
                              blockExpand->channels());

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "blockexpand", 100, false, useGpu);
  }
}

TEST(Layer, maxoutLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("maxout");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 4096, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  MaxOutConfig* maxout = input->mutable_maxout_conf();
  ImageConfig* image = maxout->mutable_image_conf();

  image->set_img_size(32);
  image->set_img_size_y(32);
  image->set_channels(4);
  maxout->set_groups(2);

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "maxout", 10, false, useGpu);
  }
}

void testFcLayer(string format, size_t nnz) {
  TestConfig config;
  config.biasSize = 1024;
  config.layerConfig.set_type("fc");
  config.layerConfig.set_size(1024);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_drop_rate(0.1);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", 2048, nnz, ParaSparse(format)});
  config.layerConfig.add_inputs();

  LOG(INFO) << config.inputDefs[0].sparse.sparse << " "
            << config.inputDefs[0].sparse.format;

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "fc",
                  100,
                  /* trans */ false,
                  useGpu,
                  /* weight */ true);
  }
}

TEST(Layer, fcLayer) {
  testFcLayer("", 1024 * 1024 * 2);
  testFcLayer("csc", 1024 * 10);
  testFcLayer("csr", 1024 * 10);
}

TEST(Layer, SelectiveFullyConnectedLayer) {
  TestConfig config;
  size_t nin = 16;
  size_t nout = 256;
  config.layerConfig.set_type("selective_fc");
  config.layerConfig.set_size(nout);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_has_selected_colums(true);
  config.layerConfig.set_selective_fc_pass_generation(false);
  config.biasSize = nout;

  config.inputDefs.push_back({INPUT_DATA, "input0", nin, nin * nout});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back(
      {INPUT_SPARSE_NON_VALUE_DATA, "index", nout, 0, ParaSparse("csr", true)});
  config.layerConfig.add_inputs();

  testLayerGrad(config,
                "selective_fc",
                100,
                /* trans= */ false,
                /* useGup= */ false,
                false);
#ifdef PADDLE_WITH_CUDA
  testLayerGrad(config,
                "selective_fc",
                100,
                /* trans= */ false,
                /* useGup= */ true,
                false);
#endif
}

TEST(Layer, DataNormLayer) {
  TestConfig config;
  config.layerConfig.set_type("data_norm");
  config.layerConfig.set_size(20);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 20, 100});
  config.inputDefs.back().isStatic = true;
  config.layerConfig.add_inputs();

  for (auto strategy : {"z-score", "min-max", "decimal-scaling"}) {
    config.layerConfig.set_data_norm_strategy(strategy);
    // The parameters are static, so not support GPU now
    testLayerGrad(config,
                  "data_norm",
                  200,
                  /* trans */ false,
                  /* useGpu */ false);
  }
}

TEST(Layer, hsigmoidLayer) {
  TestConfig config;
  config.layerConfig.set_type("hsigmoid");
  config.layerConfig.set_num_classes(5);
  config.layerConfig.set_size(1);
  config.biasSize = config.layerConfig.num_classes() - 1;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 200});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 5, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "hsigmoid",
                  100,
                  /* trans */ false,
                  /* useGpu */ useGpu);
  }
}

TEST(Layer, multi_cross) {
  TestConfig config;
  config.layerConfig.set_type("multi-class-cross-entropy");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(
        config, "multi-class-cross-entropy", 100, /* trans */ false, useGpu);
  }
}

TEST(Layer, multi_binary_label_sparse_mat) {
  TestConfig config;
  config.layerConfig.set_type("multi_binary_label_cross_entropy");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_SPARSE_NON_VALUE_DATA, "layer_1", 50, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "multi_binary_label_cross_entropy",
                  100,
                  /* trans */ false,
                  useGpu);
  }
}

TEST(layer, multi_binary_label_id) {
  TestConfig config;
  config.layerConfig.set_type("multi_binary_label_cross_entropy");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "multi_binary_label_cross_entropy",
                  100,
                  /* trans */ false,
                  useGpu);
  }
}

TEST(Layer, multi_cross_with_selfnorm) {
  TestConfig config;
  config.layerConfig.set_type("multi_class_cross_entropy_with_selfnorm");
  config.layerConfig.set_softmax_selfnorm_alpha(0.1);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // Not support GPU now
  testLayerGrad(config,
                "multi_class_cross_entropy_with_selfnorm",
                100,
                /* trans */ false,
                /* useGpu */ false);
}

TEST(Layer, multi_cross_soft) {
  TestConfig config;
  config.layerConfig.set_type("soft_binary_class_cross_entropy");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "soft_binary_class_cross_entropy",
                  100,
                  /* trans */ false,
                  useGpu);
  }
}

TEST(Layer, square_error) {
  TestConfig config;
  config.layerConfig.set_type("square_error");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "square_error", 100, /* trans */ false, useGpu);
  }
}

TEST(Layer, sparse_square_error) {
  TestConfig config;
  config.layerConfig.set_type("square_error");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_SPARSE_NON_VALUE_DATA, "layer_1", 50, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // "GpuSparseMatrix" as label is not supported
  testLayerGrad(config,
                "square_error",
                100,
                /* trans */ false,
                /* useGpu */ false);
}

TEST(Layer, sparse_float_square_error) {
  TestConfig config;
  config.layerConfig.set_type("square_error");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_SPARSE_FLOAT_VALUE_DATA, "layer_1", 50, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // "GpuSparseMatrix" as label is not supported
  testLayerGrad(config,
                "square_error",
                100,
                /* trans */ false,
                /* useGpu */ false);
}

TEST(Layer, square_error_weighted) {
  TestConfig config;
  config.layerConfig.set_type("square_error");
  config.biasSize = 0;
  config.testAccumulate = false;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_1", 10, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_2", 1, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "square_error", 100, /* trans */ false, useGpu);
  }
}

TEST(Layer, huber_regression_loss) {
  TestConfig config;
  config.layerConfig.set_type("huber_regression");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    for (auto delta : {1, 3, 5}) {
      config.layerConfig.set_delta(delta);
      testLayerGrad(config, "huber_regression", 100, /* trans */ false, useGpu);
    }
  }
}

TEST(Layer, huber_two_class) {
  TestConfig config;
  config.layerConfig.set_type("huber_classification");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 2, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "huber_two_class", 100, /* trans */ false, useGpu);
  }
}

void testExpandLayer(string trans_type, bool hasSubseq) {
  TestConfig config;
  config.layerConfig.set_type("expand");

  config.inputDefs.push_back(
      {trans_type == "non-seq" ? INPUT_DENSE_DIM_DATA : INPUT_SEQUENCE_DATA,
       "layer_0",
       10,
       0});
  config.inputDefs.push_back(
      {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA,
       "layer_1",
       10,
       0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.set_trans_type(trans_type);
  LOG(INFO) << " trans_type=" << trans_type << " hasSubseq=" << hasSubseq;

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "expand", 30, false, useGpu);
  }
}

TEST(Layer, ExpandLayer) {
  testExpandLayer("non-seq", false);  // non-seq expand to seq
  testExpandLayer("non-seq", true);   // non-seq expand to hasSubseq
  testExpandLayer("seq", true);       // seq expand to hasSubseq
}

void testDegradeLayer(bool hasSubseq,
                      string layer_type,
                      string trans_type,
                      int stride) {
  TestConfig config;
  config.layerConfig.set_type(layer_type);
  config.layerConfig.set_size(10);
  config.layerConfig.set_seq_pool_stride(stride);
  config.biasSize = 0;

  config.inputDefs.push_back(
      {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA,
       "layer_0",
       10,
       0});
  config.layerConfig.add_inputs();
  config.layerConfig.set_trans_type(trans_type);

  auto testDegradeLayerGrad = [](TestConfig& config, string layer_type) {
    for (auto useGpu : {false, true}) {
      testLayerGrad(config, layer_type, 100, false, useGpu);
    }
  };

  if (layer_type == "average") {
    for (auto strategy : {"average", "sum", "squarerootn"}) {
      LOG(INFO) << " hasSubseq=" << hasSubseq << " trans_type=" << trans_type
                << " average_strategy=" << strategy
                << " seq_pool_stride=" << stride;
      config.layerConfig.set_average_strategy(strategy);
      testDegradeLayerGrad(config, layer_type);
    }
  } else {
    LOG(INFO) << " hasSubseq=" << hasSubseq << " trans_type=" << trans_type
              << " seq_pool_stride=" << stride;
    testDegradeLayerGrad(config, layer_type);
  }
}

TEST(Layer, MaxLayer) {
  testDegradeLayer(false, "max", "non-seq", -1);  // seq max to non-seq
  testDegradeLayer(false,
                   "max",
                   "non-seq",
                   5);  // seq max to a shorten seq, stride window = 5
  testDegradeLayer(true, "max", "non-seq", -1);  // hasSubseq max to non-seq
  testDegradeLayer(true, "max", "seq", -1);      // hasSubseq max to seq
}

TEST(Layer, SequenceLastInstanceLayer) {
  testDegradeLayer(false,
                   "seqlastins",
                   "non-seq",
                   -1);  // seq seqlastins to non-seq
  testDegradeLayer(false,
                   "seqlastins",
                   "non-seq",
                   5);  // seq seqlastins to a shorten seq, stride window = 5
  testDegradeLayer(true,
                   "seqlastins",
                   "non-seq",
                   -1);  // hasSubseq seqlastins to non-seq
  testDegradeLayer(true,
                   "seqlastins",
                   "seq",
                   -1);  // hasSubseq seqlastins to seq
}

TEST(Layer, AverageLayer) {
  testDegradeLayer(false, "average", "non-seq", -1);  // seq average to non-seq
  testDegradeLayer(false,
                   "average",
                   "non-seq",
                   5);  // seq average to a shorten seq, stride window = 5
  testDegradeLayer(true,
                   "average",
                   "non-seq",
                   -1);                          // hasSubseq average to non-seq
  testDegradeLayer(true, "average", "seq", -1);  // hasSubseq average to seq
}

TEST(Layer, SequenceConcatLayer) {
  TestConfig config;
  config.layerConfig.set_type("seqconcat");
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_0", 10, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "seqconcat", 100, false, useGpu);
  }
}

TEST(Layer, SequenceReshapeLayer) {
  TestConfig config;
  config.layerConfig.set_type("seqreshape");
  config.layerConfig.set_size(10);

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_0", 100, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "seqreshape", 100, false, useGpu);
  }
}

TEST(Layer, ConvShiftLayer) {
  TestConfig config;
  config.layerConfig.set_type("conv_shift");
  config.layerConfig.set_size(10);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 3, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // Not support GPU now
  testLayerGrad(config, "conv_shift", 100, false, false);
}

TEST(Layer, PowerLayer) {
  TestConfig config;
  config.layerConfig.set_type("power");
  config.layerConfig.set_size(10);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "power", 100, false, useGpu);
  }
}

TEST(Layer, ConvexCombinationLayer) {
  TestConfig config;
  config.layerConfig.set_type("convex_comb");
  config.layerConfig.set_size(20);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 5, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 100, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "convex_comb", 100, false, useGpu);
  }
}

TEST(Layer, InterpolationLayer) {
  TestConfig config;
  config.layerConfig.set_type("interpolation");
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_2", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "interpolation", 100, false, useGpu);
  }
}

TEST(Layer, DotProdLayer) {
  TestConfig config;
  config.layerConfig.set_type("dot_prod");
  config.layerConfig.set_size(1);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "dot_prod", 10, false, useGpu);
  }
}

TEST(Layer, OuterProdLayer) {
  TestConfig config;
  config.layerConfig.set_type("out_prod");
  config.layerConfig.set_size(100);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "out_prod", 100, false, useGpu);
  }
}

TEST(Layer, SlopeInterceptLayer) {
  TestConfig config;
  config.layerConfig.set_type("slope_intercept");
  config.layerConfig.set_size(10);
  config.layerConfig.set_slope(1.0);
  config.layerConfig.set_intercept(0.1);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "slope_intercept", 100, false, useGpu);
  }
}

TEST(Layer, ScalingLayer) {
  TestConfig config;
  config.layerConfig.set_type("scaling");
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 10, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "scaling", 100, false, useGpu);
  }
}

void testNormLayer(const string& normType, bool trans, bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("norm");
  config.layerConfig.set_active_type("relu");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1568, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  NormConfig* norm = input->mutable_norm_conf();
  norm->set_norm_type(normType);
  norm->set_channels(16);
  norm->set_size(5);
  norm->set_scale(0.001);
  norm->set_pow(0.75);
  norm->set_blocked(0);
  norm->set_img_size(14);
  norm->set_img_size_y(7);
  norm->set_output_x(norm->img_size());
  norm->set_output_y(norm->img_size_y());
  if (norm->norm_type() == "cmrnorm" ||
      norm->norm_type() == "cmrnorm-projection") {
    norm->set_scale(norm->scale() / norm->size());
  } else {
    norm->set_scale(norm->scale() / (norm->size() * norm->size()));
  }

  config.layerConfig.set_size(norm->output_x() * norm->output_y() *
                              norm->channels());
  config.biasSize = 0;

  testLayerGrad(config, "norm", 100, trans, useGpu);
}

TEST(Layer, NormLayer) {
  testNormLayer("cmrnorm-projection",
                /* trans= */ false, /* useGpu= */
                true);
  testNormLayer("cmrnorm-projection",
                /* trans= */ false, /* useGpu= */
                false);
}

void setPoolConfig(TestConfig* config,
                   PoolConfig* pool,
                   const string& poolType) {
  (*config).biasSize = 0;
  (*config).layerConfig.set_type("pool");
  (*config).layerConfig.set_num_filters(16);

  int kw = 3, kh = 3;
  int pw = 0, ph = 0;
  int sw = 2, sh = 2;
  pool->set_pool_type(poolType);
  pool->set_channels(16);
  pool->set_size_x(kw);
  pool->set_size_y(kh);
  pool->set_start(0);
  pool->set_padding(pw);
  pool->set_padding_y(ph);
  pool->set_stride(sw);
  pool->set_stride_y(sh);

  int ow = outputSize(pool->img_size(), kw, pw, sw, /* caffeMode */ false);
  int oh = outputSize(pool->img_size_y(), kh, ph, sh, /* caffeMode */ false);
  pool->set_output_x(ow);
  pool->set_output_y(oh);
}

void testPoolLayer(const string& poolType,
                   bool trans,
                   bool useGpu,
                   bool excludeMode = true) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3136, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  pool->set_img_size(14);
  pool->set_img_size_y(14);
  pool->set_exclude_mode(excludeMode);
  setPoolConfig(&config, pool, poolType);
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  testLayerGrad(config, "pool", 100, trans, useGpu);
}

#ifdef PADDLE_WITH_CUDA
void testPoolLayer2(const string& poolType, bool trans, bool useGpu) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3200, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  pool->set_size_y(4);
  pool->set_stride_y(3);
  pool->set_img_size(10);
  pool->set_img_size_y(20);
  setPoolConfig(&config, pool, poolType);
  pool->set_output_y((pool->img_size_y() - pool->start() - pool->size_y()) /
                         ((float)pool->stride_y()) +
                     1.5);
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  testLayerGrad(config, "pool", 100, trans, useGpu);
}
#endif

TEST(Layer, PoolLayer) {
  testPoolLayer("avg-projection", /* trans= */ false, /* useGpu= */ false);
  testPoolLayer("avg-projection",
                /* trans= */ false,
                /* useGpu= */ false,
                /* excludeMode= */ false);
  testPoolLayer("max-projection", /* trans= */ false, /* useGpu= */ false);
  testPoolLayer("max-pool-with-mask", /* trans= */ false, /* useGpu= */ false);

#ifdef PADDLE_WITH_CUDA
  testPoolLayer("avg-projection", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("avg-projection",
                /* trans= */ false,
                /* useGpu= */ true,
                /* excludeMode= */ false);
  testPoolLayer("max-projection", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("cudnn-max-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("cudnn-avg-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer2("cudnn-max-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer2("cudnn-avg-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer2("cudnn-avg-incl-pad-pool",
                 /* trans= */ false,
                 /* useGpu= */ true);
  testPoolLayer("max-pool-with-mask", /* trans= */ false, /* useGpu= */ true);
#endif
}

void setPool3DConfig(TestConfig* config,
                     PoolConfig* pool,
                     const string& poolType) {
  // filter size
  const int NUM_FILTERS = 16;
  const int FILTER_SIZE = 3;
  const int FILTER_SIZE_Y = 3;
  const int FILTER_SIZE_Z = 3;
  const int CHANNELS = 16;

  (*config).biasSize = 0;
  (*config).layerConfig.set_type("pool3d");
  (*config).layerConfig.set_num_filters(NUM_FILTERS);

  int kw = FILTER_SIZE, kh = FILTER_SIZE_Y, kd = FILTER_SIZE_Z;
  int pw = 0, ph = 0, pd = 0;
  int sw = 2, sh = 2, sd = 2;

  pool->set_pool_type(poolType);
  pool->set_pool_type("avg");
  pool->set_channels(CHANNELS);
  pool->set_size_x(kw);
  pool->set_size_y(kh);
  pool->set_size_z(kd);
  pool->set_padding(0);
  pool->set_padding_y(0);
  pool->set_padding_z(0);
  pool->set_stride(sw);
  pool->set_stride_y(sh);
  pool->set_stride_z(sd);
  pool->set_start(0);
  int ow = outputSize(pool->img_size(), kw, pw, sw, /* caffeMode */ false);
  int oh = outputSize(pool->img_size_y(), kh, ph, sh, /* caffeMode */ false);
  int od = outputSize(pool->img_size_z(), kd, pd, sd, /* caffeMode */ false);
  pool->set_output_x(ow);
  pool->set_output_y(oh);
  pool->set_output_z(od);
}

void testPool3DLayer(const string& poolType, bool trans, bool useGpu) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 11664, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  const int IMAGE_SIZE = 9;
  const int IMAGE_SIZE_Y = 9;
  const int IMAGE_SIZE_Z = 9;

  pool->set_img_size(IMAGE_SIZE);
  pool->set_img_size_y(IMAGE_SIZE_Y);
  pool->set_img_size_z(IMAGE_SIZE_Z);

  setPool3DConfig(&config, pool, poolType);
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  testLayerGrad(config, "pool3d", 100, trans, useGpu);
}

TEST(Layer, Pool3DLayer) {
  testPool3DLayer("avg", /* trans= */ false, /* useGpu= */ false);
  testPool3DLayer("max", /* trans= */ false, /* useGpu= */ false);
#ifdef PADDLE_WITH_CUDA
  testPool3DLayer("avg", /* trans= */ false, /* useGpu= */ true);
  testPool3DLayer("max", /* trans= */ false, /* useGpu= */ true);
#endif
}

void testSppLayer(const string& poolType,
                  const int pyramidHeight,
                  bool trans,
                  bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("spp");
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3200, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  SppConfig* sppConfig = input->mutable_spp_conf();
  sppConfig->set_pool_type(poolType);
  sppConfig->set_pyramid_height(pyramidHeight);
  ImageConfig* imageConfig = sppConfig->mutable_image_conf();
  imageConfig->set_channels(16);
  imageConfig->set_img_size(10);
  imageConfig->set_img_size_y(20);
  int outputSize = (std::pow(4, sppConfig->pyramid_height()) - 1) / (4 - 1);
  config.layerConfig.set_size(outputSize * imageConfig->channels());
  testLayerGrad(config, "spp", 100, trans, useGpu);
}

TEST(Layer, SpatialPyramidPoolLayer) {
  for (auto useGpu : {false, true}) {
    for (auto pyramidHeight : {1, 2, 3}) {
      testSppLayer("avg-projection", pyramidHeight, false, useGpu);
      testSppLayer("max-projection", pyramidHeight, false, useGpu);
    }
  }
}

TEST(Layer, rankCostLayer) {
  TestConfig config;
  config.layerConfig.set_type("rank-cost");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 1, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_2", 1, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "rank-cost", 100, false, useGpu);
  }
}

TEST(Layer, sumCostLayer) {
  TestConfig config;
  config.layerConfig.set_type("sum_cost");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "sum_cost", 100, false, useGpu);
  }
}

TEST(Layer, weightedRankCostLayer) {
  TestConfig config;
  config.layerConfig.set_type("rank-cost");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 1, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_2", 1, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_3", 1, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "weighted-rank-cost", 100, false, useGpu);
  }
}

TEST(Layer, TensorLayer) {
  TestConfig config;
  config.layerConfig.set_type("tensor");
  config.layerConfig.set_size(10);
  config.layerConfig.set_active_type("sigmoid");
  config.biasSize = config.layerConfig.size();

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 5, 250});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 5, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "tensor", 100, false, useGpu);
  }
}

TEST(Layer, RecurrentLayer) {
  TestConfig config;
  config.layerConfig.set_type("recurrent");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("tanh");
  config.biasSize = 4;

  config.inputDefs.push_back(
      {INPUT_SEQUENCE_DATA, "layer_0", /* dim= */ 4, /* paraSize= */ 16});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    for (auto reversed : {false, true}) {
      config.layerConfig.set_reversed(reversed);
      config.testState = !reversed;
      testLayerGrad(
          config, "recurrent", 50, /* trans= */ false, useGpu, false, 1.0);
    }
  }
}

TEST(Layer, LstmLayer) {
  TestConfig config;
  config.layerConfig.set_type("lstmemory");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("tanh");
  config.layerConfig.set_active_state_type("sigmoid");
  config.layerConfig.set_active_gate_type("sigmoid");
  config.biasSize = 28;

  config.inputDefs.push_back(
      {INPUT_SEQUENCE_DATA, "layer_0", /* dim= */ 16, /* paraSize= */ 64});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    for (auto reversed : {false, true}) {
      config.layerConfig.set_reversed(reversed);
      config.testState = !reversed;
      testLayerGrad(
          config, "lstmemory", 100, /* trans= */ false, useGpu, false, 0.02);
    }
  }
  for (auto useGpu : {true}) {
    config.testBatchState = true;
    config.layerConfig.set_reversed(false);
    testLayerGrad(config, "lstmemory", 10, /* trans= */ false, useGpu);
  }
}

TEST(Layer, MDLstmLayer) {
  TestConfig config;
  config.layerConfig.set_type("mdlstmemory");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_active_state_type("sigmoid");
  config.layerConfig.set_active_gate_type("sigmoid");
  config.biasSize = 4 * 9;

  config.inputDefs.push_back(
      {INPUT_SEQUENCE_MDIM_DATA, "layer_0", 4 * 5, 4 * 4 * 5});
  config.layerConfig.add_inputs();
  config.layerConfig.add_directions(true);
  config.layerConfig.add_directions(true);

  for (auto useGpu : {false, true}) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        config.layerConfig.set_directions(0, bool(i));
        config.layerConfig.set_directions(1, bool(j));
        testLayerGrad(config, "mdlstmemory", 100, false, useGpu);
      }
    }
  }
}

TEST(Layer, ParameterReluLayer) {
  auto testParameterReluLayer = [&](size_t inputSize, size_t channels) {
    TestConfig config;
    config.layerConfig.set_type("prelu");
    config.inputDefs.push_back({INPUT_DATA, "layer_0", inputSize, channels});
    config.layerConfig.add_inputs();
    config.layerConfig.set_size(inputSize);
    config.layerConfig.set_partial_sum(inputSize /
                                       channels);  // size of feature map
    for (auto useGpu : {false, true}) {
      testLayerGrad(config, "prelu", 100, false, useGpu);
    }
  };

  testParameterReluLayer(192, 1);
  testParameterReluLayer(192, 3);
  testParameterReluLayer(192, 192);
}

TEST(Layer, ResizeLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("resize");
  config.layerConfig.set_size(64);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 16, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "resize", 100, false, useGpu);
  }
}

TEST(Layer, RotateLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("rotate");
  const int CHANNEL = 2;
  const int HEIGHT = 8;
  const int WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;
  config.layerConfig.set_size(INPUT_SIZE);
  config.layerConfig.set_height(HEIGHT);
  config.layerConfig.set_width(WIDTH);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", INPUT_SIZE, 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "rotate", 100, false, useGpu);
  }
}

TEST(Layer, NCELayer) {
  TestConfig config;
  size_t numClasses = 4;
  config.layerConfig.set_type("nce");
  config.layerConfig.set_size(1);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_num_classes(numClasses);
  config.biasSize = numClasses;

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", /* dim= */ 16, /* paraSize= */ 16 * numClasses});
  config.inputDefs.push_back(
      {INPUT_LABEL, "label", /* dim= */ numClasses, /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto withWeight : {false, true}) {
    if (withWeight) {
      config.inputDefs.push_back(
          {INPUT_DATA_TARGET, "weight", /* dim= */ 1, /* paraSize= */ 0});
      config.layerConfig.add_inputs();
    }

    for (auto isIdLabel : {false, true}) {
      config.inputDefs[1] = {
          isIdLabel ? INPUT_LABEL : INPUT_SPARSE_NON_VALUE_DATA,
          "label",
          /* dim= */ numClasses,
          /* paraSize= */ 0};

      for (auto withDist : {false, true}) {
        config.layerConfig.clear_neg_sampling_dist();
        if (withDist) {
          double sum = 0;
          for (size_t i = 0; i < numClasses; ++i) {
            real p = rand();  // NOLINT use rand_r
            config.layerConfig.add_neg_sampling_dist(p);
            sum += p;
          }
          for (size_t i = 0; i < numClasses; ++i) {
            real p = config.layerConfig.neg_sampling_dist(i) / sum;
            config.layerConfig.set_neg_sampling_dist(i, p);
          }
        }
        LOG(INFO) << "NCELayer "
                  << " isIdLabel=" << isIdLabel << " withWeight=" << withWeight
                  << " withDist=" << withDist;
        // Not support GPU now
        testLayerGrad(config,
                      "nce",
                      100,
                      /* trans= */ false,
                      /* useGpu */ false);
      }
    }
  }
}

TEST(Layer, GatedRecurrentLayer) {
  TestConfig config;
  config.layerConfig.set_type("gated_recurrent");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_active_gate_type("sigmoid");
  config.biasSize = 12;

  config.inputDefs.push_back(
      {INPUT_SEQUENCE_DATA, "layer_0", /* dim= */ 12, /* paraSize= */ 48});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    for (auto reversed : {false, true}) {
      config.layerConfig.set_reversed(reversed);
      config.testState = !reversed;
      testLayerGrad(config, "gated_recurrent", 100, /* trans= */ false, useGpu);
    }
  }
}

TEST(Layer, GruStepLayer) {
  TestConfig config;
  config.layerConfig.set_type("gru_step");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_active_gate_type("sigmoid");
  config.biasSize = 12;

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", /* dim= */ 12, /* paraSize= */ 48});
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_1", /* dim= */ 4, /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "gruStep", 100, /* trans= */ false, useGpu);
  }
}

TEST(Layer, LstmStepLayer) {
  TestConfig config;
  config.layerConfig.set_type("lstm_step");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_active_state_type("sigmoid");
  config.layerConfig.set_active_gate_type("sigmoid");
  config.biasSize = 12;
  config.testAccumulate = false;

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", /* dim= */ 16, /* paraSize= */ 0});
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_1", /* dim= */ 4, /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "lstmStep", 100, /* trans= */ false, useGpu);
  }
}

void testBatchNormLayer(const string& type, bool trans, bool useGpu) {
  TestConfig config;
  const int CHANNELS = 10;
  const int IMG_SIZE = 16;
  const int IMG_SIZE_Y = 8;
  size_t size = CHANNELS * IMG_SIZE * IMG_SIZE_Y;
  config.layerConfig.set_type(type);
  config.layerConfig.set_size(size);
  config.layerConfig.set_active_type("sigmoid");
  config.biasSize = CHANNELS;
  config.inputDefs.push_back({INPUT_DATA,
                              "layer_0",
                              /* dim= */ size,
                              /* paraSize= */ CHANNELS});

  config.inputDefs.push_back({INPUT_DATA, "layer_1_running_mean", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;
  config.inputDefs.push_back({INPUT_DATA, "layer_2_running_var", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;

  LayerInputConfig* input = config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(CHANNELS);
  img_conf->set_img_size(IMG_SIZE);
  img_conf->set_img_size_y(IMG_SIZE_Y);

  testLayerGrad(config,
                "batch_norm",
                64,
                /* trans= */ trans,
                useGpu,
                /* useWeight */ true);
}

TEST(Layer, BatchNormalizationLayer) {
  testBatchNormLayer("batch_norm", false, false);
#ifdef PADDLE_WITH_CUDA
  testBatchNormLayer("batch_norm", false, true);
  if (hl_get_cudnn_lib_version() >= int(4000)) {
    testBatchNormLayer("cudnn_batch_norm", false, true);
  }
#endif
}

void testBatchNorm3DLayer(const string& type, bool trans, bool useGpu) {
  TestConfig config;
  const int CHANNELS = 10;
  const int IMG_SIZE = 16;
  const int IMG_SIZE_Y = 8;
  const int IMG_SIZE_Z = 8;
  size_t size = CHANNELS * IMG_SIZE * IMG_SIZE_Y * IMG_SIZE_Z;
  config.layerConfig.set_type(type);
  config.layerConfig.set_size(size);
  config.layerConfig.set_active_type("sigmoid");
  config.biasSize = CHANNELS;
  config.inputDefs.push_back({INPUT_DATA,
                              "layer_0",
                              /* dim= */ size,
                              /* paraSize= */ CHANNELS});

  config.inputDefs.push_back({INPUT_DATA, "layer_1_running_mean", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;
  config.inputDefs.push_back({INPUT_DATA, "layer_2_running_var", 1, CHANNELS});
  config.inputDefs.back().isStatic = true;

  LayerInputConfig* input = config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(CHANNELS);
  img_conf->set_img_size(IMG_SIZE);
  img_conf->set_img_size_y(IMG_SIZE_Y);
  img_conf->set_img_size_z(IMG_SIZE_Z);

  testLayerGrad(config,
                "batch_norm",
                64,
                /* trans= */ trans,
                useGpu,
                /* useWeight */ true);
}

TEST(Layer, testBatchNorm3DLayer) {
  testBatchNorm3DLayer("batch_norm", false, false);
#ifdef PADDLE_WITH_CUDA
  testBatchNorm3DLayer("batch_norm", false, true);
  if (hl_get_cudnn_lib_version() >= int(4000)) {
    testBatchNorm3DLayer("cudnn_batch_norm", false, true);
  }
#endif
}

void testConvOperator(bool isDeconv) {
  TestConfig config;
  const int NUM_FILTERS = 16;
  const int FILTER_SIZE = 2;
  const int FILTER_SIZE_Y = 3;
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 16;
  const int IMAGE_SIZE_Y = 9;
  OperatorConfig& operatorConf = *config.layerConfig.add_operator_confs();
  if (isDeconv) {
    operatorConf.set_type("convt");
  } else {
    operatorConf.set_type("conv");
  }
  ConvConfig* conv = operatorConf.mutable_conv_conf();
  operatorConf.set_num_filters(NUM_FILTERS);
  conv->set_filter_size(FILTER_SIZE);
  conv->set_filter_size_y(FILTER_SIZE_Y);
  conv->set_channels(CHANNELS);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_img_size(IMAGE_SIZE);
  conv->set_img_size_y(IMAGE_SIZE_Y);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /*  caffeMode */ true));
  conv->set_output_y(outputSize(conv->img_size_y(),
                                conv->filter_size_y(),
                                conv->padding_y(),
                                conv->stride_y(),
                                /*  caffeMode */ true));

  if (isDeconv) {
    conv->set_filter_channels(NUM_FILTERS / conv->groups());
    config.inputDefs.push_back({INPUT_DATA,
                                "layer_0",
                                conv->output_x() * conv->output_y() * CHANNELS,
                                0});
    config.layerConfig.set_size(IMAGE_SIZE * IMAGE_SIZE_Y * NUM_FILTERS);
  } else {
    conv->set_filter_channels(conv->channels() / conv->groups());
    config.inputDefs.push_back(
        {INPUT_DATA, "layer_0", IMAGE_SIZE * IMAGE_SIZE_Y * CHANNELS, 0});
    config.layerConfig.set_size(conv->output_x() * conv->output_y() *
                                NUM_FILTERS);
  }

  config.inputDefs.push_back(
      {INPUT_DATA,
       "layer_1",
       FILTER_SIZE * FILTER_SIZE_Y * CHANNELS * NUM_FILTERS,
       0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  testOperatorGrad(config, operatorConf, 100, /*useGpu*/ true, false);
}

TEST(Operator, conv) {
  testConvOperator(/*isDeconv*/ true);
  testConvOperator(/*isDeconv*/ false);
}

TEST(Layer, FeatureMapExpandLayer) {
  TestConfig config;
  config.layerConfig.set_type("featmap_expand");
  const int CHANNELS = 10;
  const int INPUT_SIZE = 100;
  config.layerConfig.set_size(INPUT_SIZE * CHANNELS);
  config.layerConfig.set_num_filters(CHANNELS);
  config.inputDefs.push_back({INPUT_SEQUENCE_DATA,
                              "layer_0",
                              /* dim= */ INPUT_SIZE,
                              /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false, true}) {
    for (auto asRowVec : {false, true}) {
      config.layerConfig.set_user_arg(asRowVec ? "as_row_vec" : "as_col_vec");
      testLayerGrad(config,
                    "featmap_expand",
                    /*batch_size*/ 100,
                    /* trans= */ false,
                    useGpu,
                    /* useWeight */ true);
    }
  }
}

TEST(Layer, MultiplexLayer) {
  TestConfig config;
  const int LAYER_SIZE = 100;
  config.layerConfig.set_type("multiplex");
  config.layerConfig.set_size(LAYER_SIZE);

  config.inputDefs.push_back({INPUT_LABEL, "layer_0", 2, 0});
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_1", /* dim= */ LAYER_SIZE, /* paraSize= */ 0});
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_2", /* dim= */ LAYER_SIZE, /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "multiplex", 512, /* trans= */ false, useGpu);
  }
}

TEST(Layer, PadLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("pad");

  int c = 4;
  int h = 31;
  int w = 36;
  size_t size = c * h * w;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", size, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PadConfig* pad = input->mutable_pad_conf();
  ImageConfig* image = pad->mutable_image_conf();

  image->set_channels(c);
  image->set_img_size(h);
  image->set_img_size_y(w);
  pad->add_pad_c(1);
  pad->add_pad_c(2);
  pad->add_pad_h(2);
  pad->add_pad_h(3);
  pad->add_pad_w(3);
  pad->add_pad_w(5);

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "pad", 10, false, useGpu);
  }
}

TEST(Layer, CrossChannelNormLayer) {
  TestConfig config;
  config.paramInitialMean = 1.;
  config.paramInitialStd = 0.;
  config.layerConfig.set_type("norm");
  config.layerConfig.set_size(100);
  LayerInputConfig* input = config.layerConfig.add_inputs();
  NormConfig* norm = input->mutable_norm_conf();
  norm->set_norm_type("cross-channel-norm");
  norm->set_channels(10);
  norm->set_size(100);
  norm->set_scale(0);
  norm->set_pow(0);
  norm->set_blocked(0);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 100, 10});

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "cross-channel-norm", 10, false, useGpu, false);
  }
}

TEST(Layer, smooth_l1) {
  TestConfig config;
  config.layerConfig.set_type("smooth_l1");

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 200, 0});
  config.inputDefs.push_back({INPUT_DATA_TARGET, "layer_1", 200, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "smooth_l1", 100, false, useGpu, false);
  }
}

TEST(Layer, multibox_loss) {
  TestConfig config;
  config.layerConfig.set_type("multibox_loss");
  config.biasSize = 0;
  LayerInputConfig* input = config.layerConfig.add_inputs();
  MultiBoxLossConfig* multiboxLoss = input->mutable_multibox_loss_conf();
  multiboxLoss->set_num_classes(21);
  multiboxLoss->set_input_num(1);
  multiboxLoss->set_overlap_threshold(0.5);
  multiboxLoss->set_neg_pos_ratio(3);
  multiboxLoss->set_neg_overlap(0.5);
  multiboxLoss->set_background_id(0);
  multiboxLoss->set_height(3);
  multiboxLoss->set_width(3);

  size_t gtNum = 1;
  MatrixPtr labelValue = Matrix::create(gtNum, 6, false, false);
  labelValue->randomizeUniform();
  labelValue->add(-0.5);
  labelValue->sigmoid(*labelValue);
  real* labelData = labelValue->getData();
  size_t labelWidth = labelValue->getWidth();
  for (size_t i = 0; i < gtNum; ++i) {
    *(labelData + i * labelWidth) = std::rand() % 20 + 1;
    *(labelData + i * labelWidth + 1) = 0.400259;
    *(labelData + i * labelWidth + 2) = 0.377857;
    *(labelData + i * labelWidth + 3) = 0.525712;
    *(labelData + i * labelWidth + 4) = 0.519368;
  }
  vector<int> seqStartPositions(gtNum + 1, 0);
  for (size_t i = 1; i <= gtNum; ++i) {
    seqStartPositions[i] = i;
  }

  // Ensure at lease one matched bbox
  MatrixPtr priorValue = Matrix::create(1, 72, false, false);
  priorValue->randomizeUniform();
  priorValue->add(-0.5);
  priorValue->sigmoid(*priorValue);
  real* priorData = priorValue->getData();
  *(priorData) = 0.424811;
  *(priorData + 1) = 0.397059;
  *(priorData + 2) = 0.538905;
  *(priorData + 3) = 0.447091;
  *(priorData + 4) = 0.425720;
  *(priorData + 5) = 0.515228;
  *(priorData + 6) = 0.519452;
  *(priorData + 7) = 0.591065;

  config.inputDefs.push_back(
      {INPUT_SELF_DEFINE_DATA, "priorbox", priorValue, {}});
  config.inputDefs.push_back(
      {INPUT_SELF_DEFINE_DATA, "label", labelValue, seqStartPositions});
  config.inputDefs.push_back({INPUT_DATA, "locPred", 36, 0});
  config.inputDefs.push_back({INPUT_DATA, "confPred", 189, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "multibox_loss", 1, false, useGpu, false);
  }
}

TEST(Layer, TransLayer) {
  TestConfig config;
  const int height = 128;
  const int width = 256;
  config.layerConfig.set_type("trans");
  config.layerConfig.set_size(width);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", /* dim= */ height * width, /* paraSize= */ 0});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "trans", height, /* trans= */ false, useGpu);
  }
}

TEST(Layer, RowConvLayer) {
  const int context = 3;
  const int size = 512;

  TestConfig config;
  config.layerConfig.set_type("row_conv");
  config.layerConfig.set_size(size);
  config.layerConfig.set_active_type("sigmoid");

  config.inputDefs.push_back(
      {INPUT_SEQUENCE_DATA, "layer_0", size, context * size});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  RowConvConfig* conv = input->mutable_row_conv_conf();
  conv->set_context_length(context);

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "row_conv", 100, false, useGpu, false);
  }
}

TEST(Layer, CropLayer) {
  TestConfig config;
  // config input_0
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1024, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ImageConfig* img = input->mutable_image_conf();
  img->set_channels(4);
  img->set_img_size(16);
  config.layerConfig.set_axis(2);
  config.layerConfig.add_offset(0);
  config.layerConfig.add_offset(0);

  // config input_1
  config.inputDefs.push_back({INPUT_DATA, "layer_1", 128, 0});
  input = config.layerConfig.add_inputs();
  img = input->mutable_image_conf();
  img->set_channels(2);
  img->set_img_size(8);

  // config crop layer
  config.layerConfig.set_type("crop");
  config.layerConfig.set_name("cropLayer");

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "crop", 100, false, useGpu, false);
  }
}

TEST(Layer, roi_pool) {
  TestConfig config;
  config.layerConfig.set_type("roi_pool");
  config.biasSize = 0;
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ROIPoolConfig* roiPoolConf = input->mutable_roi_pool_conf();
  roiPoolConf->set_pooled_width(7);
  roiPoolConf->set_pooled_height(7);
  roiPoolConf->set_spatial_scale(1. / 16);
  roiPoolConf->set_width(14);
  roiPoolConf->set_height(14);

  const size_t roiNum = 10;
  const size_t roiDim = 10;
  const size_t batchSize = 5;
  MatrixPtr roiValue = Matrix::create(roiNum, roiDim, false, false);
  roiValue->zeroMem();
  real* roiData = roiValue->getData();
  for (size_t i = 0; i < roiNum; ++i) {
    roiData[i * roiDim + 0] = std::rand() % batchSize;
    roiData[i * roiDim + 1] = std::rand() % 224;  // xMin
    roiData[i * roiDim + 2] = std::rand() % 224;  // yMin
    size_t xMin = static_cast<size_t>(roiData[i * roiDim + 1]);
    size_t yMin = static_cast<size_t>(roiData[i * roiDim + 2]);
    roiData[i * roiDim + 3] = xMin + std::rand() % (224 - xMin);  // xMax
    roiData[i * roiDim + 4] = yMin + std::rand() % (224 - yMin);  // yMax
  }

  config.inputDefs.push_back({INPUT_DATA, "input", 3 * 14 * 14, {}});
  config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA, "rois", roiValue, {}});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "roi_pool", batchSize, false, useGpu, false);
  }
}

TEST(Layer, SwitchOrderLayer) {
  TestConfig config;
  // config input_0
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1024, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ImageConfig* img = input->mutable_image_conf();
  img->set_channels(4);
  img->set_img_size(16);
  img->set_img_size_y(16);

  ReshapeConfig* reshape = config.layerConfig.mutable_reshape_conf();
  reshape->add_height_axis(0);
  reshape->add_height_axis(1);
  reshape->add_height_axis(2);
  reshape->add_width_axis(3);

  // config softmax layer
  config.layerConfig.set_type("switch_order");
  config.layerConfig.set_name("switchOrderLayer");

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "switch_order", 100, false, useGpu, true);
  }
}

vector<real> randSampling(real range, int n) {
  CHECK_GE(range, n);
  vector<real> num(range);
  iota(begin(num), end(num), 0.);
  if (range == n) return num;

  random_shuffle(begin(num), end(num));
  num.resize(n);
  sort(begin(num), end(num));
  return num;
}

TEST(Layer, SubNestedSequenceLayer) {
  // layer size is not crutial for this layer,
  // so use a small layer size in unittest
  const int layerSize = 4;

  const int maxSeqNum = 50;
  const int maxSeqLen = 50;
  const int maxBeamSize = 32;

  srand((size_t)(time(NULL)));
  int beamSize = 1 + (rand() % maxBeamSize);

  TestConfig config;
  config.layerConfig.set_type("sub_nested_seq");
  config.layerConfig.set_name("sub_nested_seq_layer");
  config.layerConfig.set_size(layerSize);

  int seqNum = 1 + (rand() % maxSeqNum);

  // sequence information for the first input, it is a nested sequence
  vector<int> seqStartPos(seqNum + 1, 0);
  vector<int> subSeqStartPos(1, 0);

  // selected indices
  MatrixPtr selectedIndices = Matrix::create(seqNum, beamSize, false, false);
  selectedIndices->one();
  selectedIndices->mulScalar(-1.);
  real* indicesData = selectedIndices->getData();

  for (int i = 0; i < seqNum; ++i) {
    int subSeqNum = 1 + (rand() % maxSeqNum);
    for (int j = 0; j < subSeqNum; ++j) {
      subSeqStartPos.push_back(subSeqStartPos.back() +
                               (1 + (rand() % maxSeqLen)));
    }
    vector<real> selSeqs =
        randSampling(static_cast<real>(subSeqNum), min(beamSize, subSeqNum));
    memcpy(indicesData + (i * beamSize),
           selSeqs.data(),
           selSeqs.size() * sizeof(real));
    seqStartPos[i + 1] = subSeqStartPos.back();
  }

  MatrixPtr seqInputPtr =
      Matrix::create(seqStartPos.back(), layerSize, false, false);
  seqInputPtr->randomizeUniform();
  config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                              "nested_seq_input",
                              seqInputPtr,
                              seqStartPos,
                              subSeqStartPos});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back(
      {INPUT_SELF_DEFINE_DATA, "selected_indices", selectedIndices});
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  "sub_nested_seq",
                  /* batchSize */ seqNum,
                  /* trans */ false,
                  /* useGpu*/ useGpu,
                  /* useWeight */ false);
  }
}

TEST(Layer, ClipLayer) {
  const size_t batchSize = 128;
  const size_t size = 512;
  TestConfig config;
  config.layerConfig.set_type("clip");
  config.inputDefs.push_back({INPUT_DATA, "input", size, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ClipConfig* layerConf = input->mutable_clip_conf();
  double p1 = std::rand() / (double)RAND_MAX;
  double p2 = std::rand() / (double)RAND_MAX;
  layerConf->set_min(std::min(p1, p2));
  layerConf->set_max(std::max(p1, p2));
  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "clip", batchSize, false, useGpu, false);
  }
}

TEST(Layer, RowL2NormLayer) {
  const size_t batchSize = 128;
  const size_t size = 512;
  TestConfig config;
  config.layerConfig.set_type("row_l2_norm");
  config.layerConfig.set_size(size);
  config.inputDefs.push_back({INPUT_DATA, "input", size, 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "row_l2_norm", batchSize, false, useGpu, false);
  }
}

void test3DConvLayer(const string& type, bool trans, bool useGpu) {
  // filter size
  const int NUM_FILTERS = 6;
  // const int CHANNELS = 3;
  const int FILTER_SIZE = 3;
  const int FILTER_SIZE_Y = 3;
  const int FILTER_SIZE_Z = 3;

  // input image
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 9;
  const int IMAGE_SIZE_Y = 9;
  const int IMAGE_SIZE_Z = 9;

  TestConfig config;
  config.biasSize = NUM_FILTERS;
  config.layerConfig.set_type(type);
  config.layerConfig.set_num_filters(NUM_FILTERS);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  // Setting up conv3D-trans layer
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();

  conv->set_channels(CHANNELS);
  conv->set_filter_size(FILTER_SIZE);
  conv->set_filter_size_y(FILTER_SIZE_Y);
  conv->set_filter_size_z(FILTER_SIZE_Z);
  conv->set_padding(0);
  conv->set_padding_y(0);
  conv->set_padding_z(0);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_stride_z(2);
  conv->set_img_size(IMAGE_SIZE);
  conv->set_img_size_y(IMAGE_SIZE_Y);
  conv->set_img_size_z(IMAGE_SIZE_Z);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /*  caffeMode */ true));
  conv->set_output_y(outputSize(conv->img_size_y(),
                                conv->filter_size_y(),
                                conv->padding_y(),
                                conv->stride_y(),
                                /*  caffeMode */ true));
  conv->set_output_z(outputSize(conv->img_size_z(),
                                conv->filter_size_z(),
                                conv->padding_z(),
                                conv->stride_z(),
                                /*  caffeMode */ true));

  config.layerConfig.set_size(conv->output_x() * conv->output_y() *
                              conv->output_z() * NUM_FILTERS);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  config.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       CHANNELS * IMAGE_SIZE * IMAGE_SIZE_Y * IMAGE_SIZE_Z,
       conv->filter_channels() * FILTER_SIZE * FILTER_SIZE_Y * FILTER_SIZE_Z *
           NUM_FILTERS});

  testLayerGrad(config, "conv3D", 10, trans, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "conv3D", 2, trans, useGpu, true, 0.02);
}

TEST(Layer, test3DConvLayer) {
  test3DConvLayer("conv3d", /* trans= */ false, /* useGpu= */ false);
#ifdef PADDLE_WITH_CUDA
  test3DConvLayer("conv3d", /* trans= */ false, /* useGpu= */ true);
#endif
}

void test3DDeConvLayer(const string& type, bool trans, bool useGpu) {
  // filter size
  const int NUM_FILTERS = 6;
  // const int CHANNELS = 3;
  const int FILTER_SIZE = 3;
  const int FILTER_SIZE_Y = 3;
  const int FILTER_SIZE_Z = 3;

  // input image
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 4;
  const int IMAGE_SIZE_Y = 6;
  const int IMAGE_SIZE_Z = 6;

  // Setting up conv-trans layer
  TestConfig config;
  config.biasSize = NUM_FILTERS;
  config.layerConfig.set_type("deconv3d");
  config.layerConfig.set_num_filters(NUM_FILTERS);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();

  conv->set_channels(CHANNELS);
  conv->set_filter_size(FILTER_SIZE);
  conv->set_filter_size_y(FILTER_SIZE_Y);
  conv->set_filter_size_z(FILTER_SIZE_Z);
  conv->set_padding(0);
  conv->set_padding_y(0);
  conv->set_padding_z(0);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_stride_z(2);
  conv->set_output_x(IMAGE_SIZE);
  conv->set_output_y(IMAGE_SIZE_Y);
  conv->set_output_z(IMAGE_SIZE_Z);

  conv->set_img_size(imageSize(conv->output_x(),
                               conv->filter_size(),
                               conv->padding(),
                               conv->stride(),
                               true));
  conv->set_img_size_y(imageSize(conv->output_y(),
                                 conv->filter_size_y(),
                                 conv->padding_y(),
                                 conv->stride_y(),
                                 true));
  conv->set_img_size_z(imageSize(conv->output_z(),
                                 conv->filter_size_z(),
                                 conv->padding_z(),
                                 conv->stride_z(),
                                 true));
  config.layerConfig.set_size(conv->img_size() * conv->img_size_y() *
                              conv->img_size_z() * NUM_FILTERS);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  config.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       CHANNELS * IMAGE_SIZE * IMAGE_SIZE_Y * IMAGE_SIZE_Z,
       conv->filter_channels() * FILTER_SIZE * FILTER_SIZE_Y * FILTER_SIZE_Z *
           NUM_FILTERS});

  testLayerGrad(config, "deconv3D", 10, trans, useGpu);
  // Use small batch_size and useWeight=true to test biasGrad
  testLayerGrad(config, "deconv3D", 2, trans, useGpu, true, 0.02);
}

TEST(Layer, test3DDeConvLayer) {
  test3DDeConvLayer("deconv3d", /* trans= */ false, /* useGpu= */ false);
#ifdef PADDLE_WITH_CUDA
  test3DDeConvLayer("deconv3d", /* trans= */ false, /* useGpu= */ true);
#endif
}

TEST(Layer, ScaleShiftLayer) {
  // FIXME: Disable ScaleShiftLayer because it is not stable.
  // https://github.com/PaddlePaddle/Paddle/issues/7781
  return;
  //  const size_t batchSize = 16;
  //  const size_t size = 32;
  //  TestConfig config;
  //  config.layerConfig.set_type("scale_shift");
  //  config.layerConfig.set_size(size);
  //  config.biasSize = 1;
  //  config.inputDefs.push_back(
  //      {INPUT_DATA, "input", /* dim= */ size, /* paraSize= */ 1});
  //  config.layerConfig.add_inputs();
  //  for (auto useGpu : {false, true}) {
  //    testLayerGrad(config, "scale_shift", batchSize, false, useGpu, false);
  //  }
}

TEST(Layer, ScaleSubRegionLayer) {
  const size_t batchSize = 64;
  const size_t size = 4096;
  TestConfig config;
  config.layerConfig.set_type("scale_sub_region");
  config.inputDefs.push_back({INPUT_DATA, "input", size, 0});
  MatrixPtr indicesV = Matrix::create(batchSize, 6, false, false);
  auto* data = indicesV->getData();
  for (size_t i = 0; i < batchSize; ++i) {
    data[i * 2] = 2;
    data[i * 2 + 1] = 4;
    data[i * 2 + 2] = 16;
    data[i * 2 + 3] = 32;
    data[i * 2 + 4] = 16;
    data[i * 2 + 5] = 32;
  }
  config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA, "indices", indicesV, {}});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ScaleSubRegionConfig* scaleSubRegionConf =
      input->mutable_scale_sub_region_conf();
  ImageConfig* imgConf = scaleSubRegionConf->mutable_image_conf();
  imgConf->set_img_size(32);
  imgConf->set_img_size_y(32);
  imgConf->set_channels(4);
  scaleSubRegionConf->set_value(2.0);
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "scale_sub_region", batchSize, false, useGpu, false);
  }
}

TEST(Layer, L2DistanceLayer) {
  TestConfig config;
  config.layerConfig.set_type("l2_distance");
  config.layerConfig.set_size(1);
  config.biasSize = 0;

  const size_t input_dim = 27;
  const size_t batch_size = 11;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", input_dim, 0});
  config.inputDefs.push_back({INPUT_DATA, "layer_1", input_dim, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "l2_distance", batch_size, false, useGpu);
  }
}

void testFactorizationMachineLayer(InputType type, bool useGpu) {
  const int FACTOR_SIZE = 10;
  TestConfig config;
  config.layerConfig.set_type("factorization_machine");
  config.layerConfig.set_factor_size(FACTOR_SIZE);
  config.layerConfig.set_size(1);
  config.biasSize = 0;
  config.inputDefs.push_back({type, "layer_0", 128, 1280});
  config.layerConfig.add_inputs();
  testLayerGrad(config, "factorization_machine", 16, false, useGpu, false);
}

TEST(Layer, FactorizationMachineLayer) {
  for (auto useGpu : {false, true}) {
    testFactorizationMachineLayer(INPUT_DATA, useGpu);
  }
  testFactorizationMachineLayer(INPUT_SPARSE_FLOAT_VALUE_DATA, false);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
