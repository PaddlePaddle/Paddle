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

#include <gtest/gtest.h>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/testing/TestUtil.h"
#include "paddle/trainer/Trainer.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

enum InputType {
  INPUT_DATA,         // dense vector
  INPUT_LABEL,        // id
  INPUT_DATA_TARGET,  // dense vector, but no gradient
  INPUT_SEQUENCE_DATA,
  INPUT_SEQUENCE_LABEL,
  INPUT_SPARSE_NON_VALUE_DATA
};

struct InputDef {
  InputType inputType;
  string name;
  size_t dim;
};

struct TestConfig {
  EvaluatorConfig evaluatorConfig;
  std::vector<InputDef> inputDefs;
  bool testAccumulate;
  TestConfig() : testAccumulate(true) {}
};

void testEvaluator(TestConfig testConf,
                   string testEvaluatorName,
                   size_t batchSize,
                   bool useGpu) {
#ifndef PADDLE_WITH_CUDA
  if (useGpu) return;
#endif
  FLAGS_use_gpu = useGpu;
  testConf.evaluatorConfig.set_name(testEvaluatorName);
  LOG(INFO) << " evaluator_type=" << testConf.evaluatorConfig.type()
            << " useGpu=" << useGpu;

  std::vector<Argument> arguments;
  for (size_t i = 0; i < testConf.inputDefs.size(); ++i) {
    Argument data;
    size_t dim = testConf.inputDefs[i].dim;
    switch (testConf.inputDefs[i].inputType) {
      case INPUT_DATA:
      case INPUT_SEQUENCE_DATA:
      case INPUT_DATA_TARGET:
        data.value = Matrix::create(batchSize, dim, false, useGpu);
        data.value->randomizeUniform();

        // make sure output > 0 && output < 1
        data.value->add(-0.5);
        data.value->sigmoid(*data.value);
        break;
      case INPUT_LABEL:
      case INPUT_SEQUENCE_LABEL:
        data.ids = VectorT<int>::create(batchSize, useGpu);
        data.ids->rand(dim);  // now rand number can be 0 to inputDefs[i].dim.
        break;
      case INPUT_SPARSE_NON_VALUE_DATA:
        data.value = makeRandomSparseMatrix(batchSize,
                                            dim,
                                            /* withValue= */ false,
                                            useGpu);
        break;
      default:
        LOG(FATAL) << " unknown inputType ";
        return;
    }

    ICpuGpuVectorPtr sequenceStartPositions;
    if (testConf.inputDefs[i].inputType == INPUT_SEQUENCE_DATA ||
        testConf.inputDefs[i].inputType == INPUT_SEQUENCE_LABEL) {
      if (!sequenceStartPositions) {
        generateSequenceStartPositions(batchSize, sequenceStartPositions);
      }
      data.sequenceStartPositions = sequenceStartPositions;
    }

    arguments.push_back(data);
  }

  Evaluator* testEvaluator = Evaluator::create(testConf.evaluatorConfig);
  double totalScore = 0.0;
  testEvaluator->start();
  totalScore += testEvaluator->evalImp(arguments);
  testEvaluator->updateSamplesNum(arguments);
  testEvaluator->finish();
  LOG(INFO) << *testEvaluator;

  std::vector<std::string> names;
  testEvaluator->getNames(&names);
  paddle::Error err;
  for (auto& name : names) {
    auto value = testEvaluator->getValue(name, &err);
    ASSERT_TRUE(err.isOK());
    LOG(INFO) << name << " " << value;
    auto tp = testEvaluator->getType(name, &err);
    ASSERT_TRUE(err.isOK());
    ASSERT_EQ(testConf.evaluatorConfig.type(), tp);
  }

  double totalScore2 = 0.0;
  if (testConf.testAccumulate) {
    testEvaluator->start();
    totalScore2 += testEvaluator->evalImp(arguments);
    testEvaluator->finish();
    EXPECT_LE(fabs(totalScore - totalScore2), 1.0e-5);
  }
}

void testEvaluatorAll(TestConfig testConf,
                      string testEvaluatorName,
                      size_t batchSize) {
  testEvaluator(testConf, testEvaluatorName, batchSize, true);
  testEvaluator(testConf, testEvaluatorName, batchSize, false);
}

TEST(Evaluator, detection_map) {
  TestConfig config;
  config.evaluatorConfig.set_type("detection_map");
  config.evaluatorConfig.set_overlap_threshold(0.5);
  config.evaluatorConfig.set_background_id(0);
  config.evaluatorConfig.set_ap_type("Integral");
  config.evaluatorConfig.set_evaluate_difficult(0);

  config.inputDefs.push_back({INPUT_DATA, "output", 7});
  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "label", 6});
  config.evaluatorConfig.set_evaluate_difficult(false);
  testEvaluatorAll(config, "detection_map", 100);

  config.evaluatorConfig.set_evaluate_difficult(true);
  testEvaluatorAll(config, "detection_map", 100);
}

TEST(Evaluator, classification_error) {
  TestConfig config;
  config.evaluatorConfig.set_type("classification_error");
  config.evaluatorConfig.set_top_k(5);

  config.inputDefs.push_back({INPUT_DATA, "output", 50});
  config.inputDefs.push_back({INPUT_LABEL, "label", 50});
  testEvaluatorAll(config, "classification_error", 100);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "classification_error_weight", 100);

  // multi binary labels
  config.inputDefs.clear();
  config.inputDefs.push_back({INPUT_DATA, "output", 100});
  config.inputDefs.push_back({INPUT_SPARSE_NON_VALUE_DATA, "label", 100});
  // Not support GPU
  testEvaluator(config, "classification_error_multi_binary_label", 50, false);

  config.evaluatorConfig.set_classification_threshold(0.4);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  // Not support GPU
  testEvaluator(
      config, "classification_error_weight_multi_binary_label", 50, false);
}

TEST(Evaluator, sum) {
  TestConfig config;
  config.evaluatorConfig.set_type("sum");

  // sum of output
  config.inputDefs.push_back({INPUT_DATA, "output", 10});
  testEvaluatorAll(config, "sum_output", 200);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "sum_output_weight", 200);

  // sum of label
  config.inputDefs.clear();
  config.inputDefs.push_back({INPUT_LABEL, "label", 10});
  testEvaluatorAll(config, "sum_label", 200);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "sum_label_weight", 200);
}

TEST(Evaluator, last_column_sum) {
  TestConfig config;
  config.evaluatorConfig.set_type("last-column-sum");

  config.inputDefs.push_back({INPUT_DATA, "output", 50});
  testEvaluatorAll(config, "last-column-sum", 200);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "last-column-sum_weight", 200);
}

TEST(Evaluator, last_column_auc) {
  TestConfig config;
  config.evaluatorConfig.set_type("last-column-auc");

  config.inputDefs.push_back({INPUT_DATA, "output", 2});
  config.inputDefs.push_back({INPUT_LABEL, "label", 2});
  testEvaluatorAll(config, "last-column-auc", 500);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "last-column-auc_weight", 200);
}

TEST(Evaluator, precision_recall) {
  TestConfig config;
  config.evaluatorConfig.set_type("precision_recall");

  config.inputDefs.push_back({INPUT_DATA, "output", 10});
  config.inputDefs.push_back({INPUT_LABEL, "label", 10});
  testEvaluatorAll(config, "precision_recall", 200);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  testEvaluatorAll(config, "precision_recall_weight", 200);

  LOG(INFO) << "positive_label = 5";
  config.evaluatorConfig.set_positive_label(5);
  testEvaluatorAll(config, "precision_recall_weight", 200);

  // multi binary labels
  config.inputDefs.clear();
  config.evaluatorConfig.set_positive_label(-1);
  config.inputDefs.push_back({INPUT_DATA, "output", 10});
  config.inputDefs.push_back({INPUT_SPARSE_NON_VALUE_DATA, "label", 10});
  // Not support GPU
  testEvaluator(config, "precision_recall_multi_binary_label", 100, false);

  LOG(INFO) << "classification_threshold = 0.4";
  config.evaluatorConfig.set_classification_threshold(0.4);
  config.inputDefs.push_back({INPUT_DATA, "weight", 1});
  // Not support GPU
  testEvaluator(
      config, "precision_recall_weight_multi_binary_label", 100, false);
}

TEST(Evaluator, ctc_error_evaluator) {
  TestConfig config;
  config.evaluatorConfig.set_type("ctc_edit_distance");

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA, "output", 32});
  config.inputDefs.push_back({INPUT_SEQUENCE_LABEL, "label", 1});
  testEvaluatorAll(config, "ctc_error_evaluator", 100);
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
