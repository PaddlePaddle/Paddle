/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include <vector>
#include <string>
#include "paddle/gserver/layers/DataLayer.h"
#include "ModelConfig.pb.h"
#include "paddle/trainer/Trainer.h"

#include "TestUtil.h"
#include "LayerGradUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_bool(use_gpu);
P_DECLARE_int32(gpu_id);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);

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
      for (auto batchSize : {1, 2, 5, 20, 100}) {
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
                conf, INPUT_SEQUENCE_DATA,
                trainablePadding ? conf.input_size() * pad : 0, batchSize,
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
    testProjectionGrad(conf, INPUT_DATA, /* parameterSize */ 1000,
                       /* batchSize */ 100, useGpu);
  }
}

TEST(Projection, fc) {
  ProjectionConfig conf;
  conf.set_type("fc");
  conf.set_input_size(10);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf, INPUT_DATA, /* parameterSize */ 200,
                       /* batchSize */ 100, useGpu);
  }
}

TEST(Projection, dot_mul) {
  ProjectionConfig conf;
  conf.set_type("dot_mul");
  conf.set_input_size(20);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf, INPUT_DATA, /* parameterSize */ 20,
                       /* batchSize */ 100, useGpu);
  }
}

TEST(Projection, table) {
  ProjectionConfig conf;
  conf.set_type("table");
  conf.set_input_size(10);
  conf.set_output_size(20);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf, INPUT_LABEL, /* parameterSize */ 200,
                       /* batchSize */ 100, useGpu);
  }
}

TEST(Projection, identity) {
  ProjectionConfig conf;
  conf.set_type("identity");
  conf.set_input_size(10);
  conf.set_output_size(10);
  for (auto useGpu : {false, true}) {
    testProjectionGrad(conf, INPUT_DATA, /* parameterSize */ 0,
                       /* batchSize */ 100, useGpu);
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

TEST(Layer, CRFLayer) {
  TestConfig config;
  config.layerConfig.set_type("crf");
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_0", 10, 120});
  config.inputDefs.push_back({INPUT_SEQUENCE_LABEL, "layer_1", 10, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // Not support GPU now
  testLayerGrad(config, "crf", 100, /* trans */ false, /* useGpu */ false,
                false /*useWeight*/, 0.03 /*epsilon*/);
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
    testLayerGrad(config, "ctc", 100, /* trans */ false, /* useGpu */ useGpu);
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

void testConvLayer(const string& type, bool trans, bool useGpu) {
  TestConfig config;
  config.biasSize = 16;
  config.layerConfig.set_type(type);
  config.layerConfig.set_num_filters(16);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 768, 288});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(3);
  conv->set_channels(3);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(16);
  conv->set_output_x(
      (2 * conv->padding() + conv->img_size() - conv->filter_size()) /
          ((float)conv->stride()) +
      1.5);
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());

  testLayerGrad(config, "conv", 100, trans, useGpu);
}

TEST(Layer, convLayer) {
  testConvLayer("exconv", /* trans= */ false, /* useGpu= */ false);
#ifndef PADDLE_ONLY_CPU
  testConvLayer("exconv", /* trans= */ false, /* useGpu= */ true);
  testConvLayer("cudnn_conv", /* trans= */ false, /* useGpu= */ true);
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
  blockExpand->set_output_x(
      1 +
      (2 * blockExpand->padding_x() + blockExpand->img_size_x() -
       blockExpand->block_x() + blockExpand->stride_x() - 1) /
          blockExpand->stride_x());
  blockExpand->set_output_y(
      1 +
      (2 * blockExpand->padding_y() + blockExpand->img_size_y() -
       blockExpand->block_y() + blockExpand->stride_y() - 1) /
          blockExpand->stride_y());
  config.layerConfig.set_size(blockExpand->block_x() * blockExpand->block_y() *
                              blockExpand->channels());

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "blockexpand", 100, false, useGpu);
  }
}

void testFcLayer(string format, size_t nnz) {
  TestConfig config;
  config.biasSize = 4096;
  config.layerConfig.set_type("fc");
  config.layerConfig.set_size(4096);
  config.layerConfig.set_active_type("sigmoid");
  config.layerConfig.set_drop_rate(0.1);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", 8192, nnz, ParaSparse(format)});
  config.layerConfig.add_inputs();

  LOG(INFO) << config.inputDefs[0].sparse.sparse << " "
            << config.inputDefs[0].sparse.format;

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "fc", 100, /* trans */ false, useGpu,
                  /* weight */ true);
  }
}

TEST(Layer, fcLayer) {
  testFcLayer("", 4096 * 4096 * 2);
  testFcLayer("csc", 4096 * 40);
  testFcLayer("csr", 4096 * 40);
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

  testLayerGrad(config, "selective_fc", 100,
                /* trans= */ false, /* useGup= */ false, false);
#ifndef PADDLE_ONLY_CPU
  testLayerGrad(config, "selective_fc", 100,
                /* trans= */ false, /* useGup= */ true, false);
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
    testLayerGrad(config, "data_norm", 200, /* trans */ false,
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

  // Not support GPU now
  testLayerGrad(config, "hsigmoid", 100, /* trans */ false, /* useGpu */ false);
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
    testLayerGrad(config, "multi-class-cross-entropy", 100, /* trans */ false,
                  useGpu);
  }
}

TEST(Layer, multi_binary_label) {
  TestConfig config;
  config.layerConfig.set_type("multi_binary_label_cross_entropy");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 50, 0});
  config.inputDefs.push_back({INPUT_SPARSE_NON_VALUE_DATA, "layer_1", 50, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  // Not support GPU now
  testLayerGrad(config, "multi_binary_label_cross_entropy", 100,
                /* trans */ false, /* useGpu */ false);
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
  testLayerGrad(config, "multi_class_cross_entropy_with_selfnorm", 100,
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
    testLayerGrad(config, "soft_binary_class_cross_entropy", 100,
                  /* trans */ false, useGpu);
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
  testLayerGrad(config, "square_error", 100, /* trans */ false,
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
  testLayerGrad(config, "square_error", 100, /* trans */ false,
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

TEST(Layer, huber_two_class) {
  TestConfig config;
  config.layerConfig.set_type("huber");
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 1, 0});
  config.inputDefs.push_back({INPUT_LABEL, "layer_1", 2, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "huber", 100, /* trans */ false, useGpu);
  }
}

void testExpandLayer(string trans_type, bool hasSubseq) {
  TestConfig config;
  config.layerConfig.set_type("expand");

  config.inputDefs.push_back(
      {trans_type == "non-seq" ? INPUT_DENSE_DIM_DATA : INPUT_SEQUENCE_DATA,
       "layer_0", 10, 0});
  config.inputDefs.push_back(
      {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA, "layer_1",
       10, 0});
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

void testDegradeLayer(bool hasSubseq, string layer_type, string trans_type) {
  TestConfig config;
  config.layerConfig.set_type(layer_type);
  config.layerConfig.set_size(10);
  config.biasSize = 0;

  config.inputDefs.push_back(
      {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA, "layer_0",
       10, 0});
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
                << " average_strategy=" << strategy;
      config.layerConfig.set_average_strategy(strategy);
      testDegradeLayerGrad(config, layer_type);
    }
  } else {
    LOG(INFO) << " hasSubseq=" << hasSubseq << " trans_type=" << trans_type;
    testDegradeLayerGrad(config, layer_type);
  }
}

TEST(Layer, MaxLayer) {
  testDegradeLayer(false, "max", "non-seq");  // seq max to non-seq
  testDegradeLayer(true, "max", "non-seq");   // hasSubseq max to non-seq
  testDegradeLayer(true, "max", "seq");       // hasSubseq max to seq
}

TEST(Layer, SequenceLastInstanceLayer) {
  testDegradeLayer(false, "seqlastins",
                   "non-seq");  // seq seqlastins to non-seq
  testDegradeLayer(true, "seqlastins",
                   "non-seq");  // hasSubseq seqlastins to non-seq
  testDegradeLayer(true, "seqlastins", "seq");  // hasSubseq seqlastins to seq
}

TEST(Layer, AverageLayer) {
  testDegradeLayer(false, "average", "non-seq");  // seq average to non-seq
  testDegradeLayer(true, "average", "non-seq");  // hasSubseq average to non-seq
  testDegradeLayer(true, "average", "seq");      // hasSubseq average to seq
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

  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3136, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  NormConfig* norm = input->mutable_norm_conf();
  norm->set_norm_type(normType);
  norm->set_channels(16);
  norm->set_size(5);
  norm->set_scale(0.001);
  norm->set_pow(0.75);
  norm->set_blocked(0);
  norm->set_img_size(14);
  norm->set_output_x(norm->img_size());
  if (norm->norm_type() == "cmrnorm" ||
      norm->norm_type() == "cmrnorm-projection") {
    norm->set_scale(norm->scale() / norm->size());
  } else {
    norm->set_scale(norm->scale() / (norm->size() * norm->size()));
  }

  config.layerConfig.set_size(norm->output_x() * norm->output_x() *
                              norm->channels());
  config.biasSize = 0;

  testLayerGrad(config, "norm", 100, trans, useGpu);
}

#ifndef PADDLE_ONLY_CPU
TEST(Layer, NormLayer) {
  testNormLayer("cmrnorm-projection", /* trans= */ false, /* useGpu= */ true);
}
#endif

void setPoolConfig(TestConfig* config, PoolConfig* pool,
                   const string& poolType) {
  (*config).biasSize = 0;
  (*config).layerConfig.set_type("pool");
  (*config).layerConfig.set_num_filters(16);
  (*config).layerConfig.set_partial_sum(1);
  (*config).layerConfig.set_shared_biases(true);

  pool->set_pool_type(poolType);
  pool->set_channels(16);
  pool->set_size_x(3);
  if (poolType == "cudnn-max-pool" || poolType == "cudnn-avg-pool") {
    pool->set_padding(0);
  } else {
    pool->set_start(0);
  }
  pool->set_stride(2);
  pool->set_output_x((pool->img_size() - pool->start() - pool->size_x()) /
                         ((float)pool->stride()) +
                     1.5);
}

void testPoolLayer(const string& poolType, bool trans, bool useGpu) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3136, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  setPoolConfig(&config, pool, poolType);
  pool->set_img_size(14);
  config.layerConfig.set_size(pool->output_x() * pool->output_x() *
                              pool->channels());

  testLayerGrad(config, "pool", 100, trans, useGpu);
}

#ifndef PADDLE_ONLY_CPU
void testPoolLayer2(const string& poolType, bool trans, bool useGpu) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 3200, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  setPoolConfig(&config, pool, poolType);
  pool->set_size_y(4);
  pool->set_stride_y(3);
  pool->set_img_size(10);
  pool->set_img_size_y(20);
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
  testPoolLayer("max-projection", /* trans= */ false, /* useGpu= */ false);

#ifndef PADDLE_ONLY_CPU
  testPoolLayer("avg-projection", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("max-projection", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("cudnn-max-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer("cudnn-avg-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer2("cudnn-max-pool", /* trans= */ false, /* useGpu= */ true);
  testPoolLayer2("cudnn-avg-pool", /* trans= */ false, /* useGpu= */ true);
#endif
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
      testLayerGrad(config, "recurrent", 50, /* trans= */ false, useGpu);
    }
  }
}

TEST(Layer, LstmLayer) {
  TestConfig config;
  config.layerConfig.set_type("lstmemory");
  config.layerConfig.set_size(4);
  config.layerConfig.set_active_type("sigmoid");
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
      testLayerGrad(config, "lstmemory", 100, /* trans= */ false, useGpu);
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
          isIdLabel ? INPUT_LABEL : INPUT_SPARSE_NON_VALUE_DATA, "label",
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
        testLayerGrad(config, "nce", 100, /* trans= */ false,
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
  config.layerConfig.set_type(type);
  config.layerConfig.set_size(CHANNELS * IMG_SIZE * IMG_SIZE);
  config.layerConfig.set_active_type("sigmoid");
  config.biasSize = CHANNELS;
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
                              /* dim= */ IMG_SIZE * IMG_SIZE * CHANNELS,
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

  testLayerGrad(config, "batch_norm", 64, /* trans= */ trans, useGpu,
                /* useWeight */ true);
}

TEST(Layer, BatchNormalizationLayer) {
  testBatchNormLayer("batch_norm", false, false);
#ifndef PADDLE_ONLY_CPU
  testBatchNormLayer("batch_norm", false, true);
  if (hl_get_cudnn_lib_version() >= int(4000)) {
    testBatchNormLayer("cudnn_batch_norm", false, true);
  }
#endif
}

TEST(Operator, conv) {
  TestConfig config;
  const int NUM_FILTERS = 16;
  const int FILTER_SIZE = 2;
  const int FILTER_SIZE_Y = 3;
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 16;
  OperatorConfig& operatorConf = *config.layerConfig.add_operator_confs();
  operatorConf.set_type("conv");
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
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(IMAGE_SIZE);
  int outputSize =
      int(1.0 * (2 * conv->padding() + conv->img_size() - conv->filter_size()) /
          conv->stride()) +
      1;
  conv->set_output_x(outputSize);
  config.layerConfig.set_size(outputSize * outputSize *
                              config.layerConfig.num_filters());
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              NUM_FILTERS);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", IMAGE_SIZE * IMAGE_SIZE * CHANNELS, 0});
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_1",
       FILTER_SIZE * FILTER_SIZE_Y * CHANNELS * NUM_FILTERS, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();

  testOperatorGrad(config, operatorConf, 100, /*useGpu*/ true, false);
}

TEST(Layer, FeatureMapExpandLayer) {
  TestConfig config;
  config.layerConfig.set_type("featmap_expand");
  const int CHANNELS = 10;
  const int INPUT_SIZE = 100;
  config.layerConfig.set_size(INPUT_SIZE * CHANNELS);
  config.layerConfig.set_num_filters(CHANNELS);
  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "layer_0",
                              /* dim= */ INPUT_SIZE, /* paraSize= */ 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "featmap_expand",
                  /*batch_size*/ 100, /* trans= */ false, useGpu,
                  /* useWeight */ true);
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



int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
