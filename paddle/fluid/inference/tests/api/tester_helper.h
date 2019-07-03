// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#ifdef WITH_GPERFTOOLS
#include <gperftools/profiler.h>
#endif
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/tests/api/config_printer.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#include "paddle/fluid/inference/utils/benchmark.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(model_name, "", "model name");
DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_string(refer_result, "", "reference result for comparison");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(warmup_batch_size, 100, "batch size for quantization warmup");
// setting iterations to 0 means processing the whole dataset
DEFINE_int32(iterations, 0, "number of batches to process");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");
DEFINE_bool(use_analysis, true,
            "Running the inference program in analysis mode.");
DEFINE_bool(record_benchmark, false,
            "Record benchmark after profiling the model");
DEFINE_double(accuracy, 1e-3, "Result Accuracy.");
DEFINE_double(quantized_accuracy, 1e-2, "Result Quantized Accuracy.");
DEFINE_bool(zero_copy, false, "Use ZeroCopy to speedup Feed/Fetch.");
DEFINE_bool(warmup, false,
            "Use warmup to calculate elapsed_time more accurately. "
            "To reduce CI time, it sets false in default.");

DECLARE_bool(profile);
DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace inference {

using paddle::framework::proto::VarType;

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

void PrintConfig(const PaddlePredictor::Config *config, bool use_analysis) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    LOG(INFO) << *analysis_config;
    return;
  }
  LOG(INFO) << analysis_config->ToNativeConfig();
}

// Compare result between two PaddleTensor
void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    size_t ref_size = VecReduceToInt(ref_out.shape);
    EXPECT_GT(size, 0UL);
    EXPECT_EQ(size, ref_size);
    EXPECT_EQ(out.dtype, ref_out.dtype);
    switch (out.dtype) {
      case PaddleDType::INT64: {
        int64_t *pdata = static_cast<int64_t *>(out.data.data());
        int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
      case PaddleDType::FLOAT32: {
        float *pdata = static_cast<float *>(out.data.data());
        float *pdata_ref = static_cast<float *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          CHECK_LE(std::abs(pdata_ref[j] - pdata[j]), FLAGS_accuracy);
        }
        break;
      }
      case PaddleDType::INT32: {
        int32_t *pdata = static_cast<int32_t *>(out.data.data());
        int32_t *pdata_ref = static_cast<int32_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
    }
  }
}

// Compare result between a PaddleTensor and a ZeroCopyTensor
void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<ZeroCopyTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    EXPECT_GT(size, 0UL);
    int ref_size = 0;  // this is the number of elements not memory size
    PaddlePlace place;
    switch (out.dtype) {
      case PaddleDType::INT64: {
        int64_t *pdata = static_cast<int64_t *>(out.data.data());
        int64_t *pdata_ref = ref_out.data<int64_t>(&place, &ref_size);
        EXPECT_EQ(size, static_cast<size_t>(ref_size));
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
      case PaddleDType::FLOAT32: {
        float *pdata = static_cast<float *>(out.data.data());
        float *pdata_ref = ref_out.data<float>(&place, &ref_size);
        EXPECT_EQ(size, ref_size);
        for (size_t j = 0; j < size; ++j) {
          CHECK_LE(std::abs(pdata_ref[j] - pdata[j]), FLAGS_accuracy);
        }
        break;
      }
      case PaddleDType::INT32: {
        int32_t *pdata = static_cast<int32_t *>(out.data.data());
        int32_t *pdata_ref = ref_out.data<int32_t>(&place, &ref_size);
        EXPECT_EQ(size, ref_size);
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
    }
  }
}

std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    return CreatePaddlePredictor<AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return CreatePaddlePredictor<NativeConfig>(native_config);
}

size_t GetSize(const PaddleTensor &out) { return VecReduceToInt(out.shape); }

std::unordered_map<std::string, int> GetFuseStatis(PaddlePredictor *predictor,
                                                   int *num_ops) {
  std::unordered_map<std::string, int> res;
  auto *analysis_predictor = static_cast<AnalysisPredictor *>(predictor);
  auto *fusion_status =
      analysis_predictor->analysis_argument().fusion_statis_ptr();
  if (!fusion_status) {
    return res;
  }
  for (auto &item : *fusion_status) {
    LOG(INFO) << "fused " << item.first << " " << item.second;
  }
  int num = 0;
  for (auto &node :
       analysis_predictor->analysis_argument().main_graph().Nodes()) {
    if (node->IsOp()) {
      ++num;
    }
  }
  *num_ops = num;
  return *fusion_status;
}

void SetFakeImageInput(std::vector<std::vector<PaddleTensor>> *inputs,
                       const std::string &dirname, bool is_combined = true,
                       std::string model_filename = "model",
                       std::string params_filename = "params",
                       const std::vector<std::string> *feed_names = nullptr,
                       const int continuous_inuput_index = 0) {
  // Set fake_image_data
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");
  std::vector<std::vector<int64_t>> feed_target_shapes = GetFeedTargetShapes(
      dirname, is_combined, model_filename, params_filename);
  std::ostringstream os;
  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    os << "feed target " << i << ": {" << feed_target_shapes[i][0];
    for (size_t j = 1; j < feed_target_shapes[i].size(); ++j) {
      os << ", " << feed_target_shapes[i][j];
    }
    os << "}\n";
  }
  LOG(INFO) << os.str();
  if (feed_names) {
    PADDLE_ENFORCE_EQ(feed_names->size(), feed_target_shapes.size());
  }
  std::vector<PaddleTensor> input_slots(feed_target_shapes.size());
  for (size_t i = 0; i < feed_target_shapes.size(); ++i) {
    const auto &feed_shape = feed_target_shapes[i];
    auto &input = input_slots[i];
    std::vector<int> shape({FLAGS_batch_size});
    for (size_t s = 1; s < feed_shape.size(); ++s) {
      shape.push_back(static_cast<int>(feed_shape[s]));
    }
    if (feed_names) {
      input.name = (*feed_names)[i];
    }
    input.shape = shape;
    input.dtype = PaddleDType::FLOAT32;
    size_t len = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                 [](int a, int b) { return a * b; });
    input.data.Resize(len * sizeof(float));
    input.lod.assign({{0, static_cast<size_t>(FLAGS_batch_size)}});
    float *input_data = static_cast<float *>(input.data.data());
    // fill input data, for profile easily, do not use random data here.
    for (size_t j = 0; j < len; ++j) {
      *(input_data + j) =
          static_cast<float>((j + continuous_inuput_index) % len) / len;
    }
  }
  (*inputs).emplace_back(input_slots);
}

void GetInputPerBatch(const std::vector<std::vector<int64_t>> &in,
                      std::vector<std::vector<int64_t>> *out,
                      std::vector<size_t> *lod, size_t batch_iter,
                      size_t batch_end) {
  lod->clear();
  lod->push_back(0);
  for (auto it = in.begin() + batch_iter; it < in.begin() + batch_end; it++) {
    out->push_back(*it);
    lod->push_back(lod->back() + (*it).size());  // calculate lod
  }
}

void ConvertPaddleTensorToZeroCopyTensor(
    PaddlePredictor *predictor, const std::vector<PaddleTensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto tensor = predictor->GetInputTensor(input.name);
    tensor->Reshape(input.shape);
    tensor->SetLoD({input.lod});
    if (input.dtype == PaddleDType::INT64) {
      ZeroCopyTensorAssignData<int64_t>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::FLOAT32) {
      ZeroCopyTensorAssignData<float>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::INT32) {
      ZeroCopyTensorAssignData<int32_t>(tensor.get(), input.data);
    } else {
      LOG(ERROR) << "unsupported feed type " << input.dtype;
    }
  }
}

void PredictionWarmUp(PaddlePredictor *predictor,
                      const std::vector<std::vector<PaddleTensor>> &inputs,
                      std::vector<std::vector<PaddleTensor>> *outputs,
                      int num_threads, int tid,
                      const VarType::Type data_type = VarType::FP32) {
  int batch_size = FLAGS_batch_size;
  LOG(INFO) << "Running thread " << tid << ", warm up run...";
  if (FLAGS_zero_copy) {
    ConvertPaddleTensorToZeroCopyTensor(predictor, inputs[0]);
  }
  outputs->resize(1);
  Timer warmup_timer;
  warmup_timer.tic();
  if (!FLAGS_zero_copy) {
    predictor->Run(inputs[0], &(*outputs)[0], batch_size);
  } else {
    predictor->ZeroCopyRun();
  }
  PrintTime(batch_size, 1, num_threads, tid, warmup_timer.toc(), 1, data_type);
  if (FLAGS_profile) {
    paddle::platform::ResetProfiler();
  }
}

void PredictionRun(PaddlePredictor *predictor,
                   const std::vector<std::vector<PaddleTensor>> &inputs,
                   std::vector<std::vector<PaddleTensor>> *outputs,
                   int num_threads, int tid,
                   const VarType::Type data_type = VarType::FP32,
                   float *sample_latency = nullptr) {
  int num_times = FLAGS_repeat;
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  LOG(INFO) << "Thread " << tid << ", number of threads " << num_threads
            << ", run " << num_times << " times...";
  Timer run_timer;
  double elapsed_time = 0;
#ifdef WITH_GPERFTOOLS
  ProfilerStart("paddle_inference.prof");
#endif
  int predicted_num = 0;
  if (!FLAGS_zero_copy) {
    for (int i = 0; i < iterations; i++) {
      run_timer.tic();
      for (int j = 0; j < num_times; j++) {
        predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
      }
      elapsed_time += run_timer.toc();

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  } else {
    for (int i = 0; i < iterations; i++) {
      ConvertPaddleTensorToZeroCopyTensor(predictor, inputs[i]);
      run_timer.tic();
      for (int j = 0; j < num_times; j++) {
        predictor->ZeroCopyRun();
      }
      elapsed_time += run_timer.toc();

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  }

#ifdef WITH_GPERFTOOLS
  ProfilerStop();
#endif

  auto batch_latency = elapsed_time / (iterations * num_times);
  PrintTime(FLAGS_batch_size, num_times, num_threads, tid, batch_latency,
            iterations, data_type);

  if (sample_latency != nullptr)
    *sample_latency = batch_latency / FLAGS_batch_size;

  if (FLAGS_record_benchmark) {
    Benchmark benchmark;
    benchmark.SetName(FLAGS_model_name);
    benchmark.SetBatchSize(FLAGS_batch_size);
    benchmark.SetLatency(batch_latency);
    benchmark.PersistToFile("benchmark_record.txt");
  }
}

void TestOneThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs, bool use_analysis = true,
    const VarType::Type data_type = VarType::FP32,
    float *sample_latency = nullptr) {
  auto predictor = CreateTestPredictor(config, use_analysis);
  if (FLAGS_warmup) {
    PredictionWarmUp(predictor.get(), inputs, outputs, 1, 0, data_type);
  }
  PredictionRun(predictor.get(), inputs, outputs, 1, 0, data_type,
                sample_latency);
}

void TestMultiThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs, int num_threads,
    bool use_analysis = true) {
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  predictors.emplace_back(CreateTestPredictor(config, use_analysis));
  for (int tid = 1; tid < num_threads; tid++) {
    predictors.emplace_back(predictors.front()->Clone());
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // Each thread should have local inputs and outputs.
      // The inputs of each thread are all the same.
      std::vector<std::vector<PaddleTensor>> outputs_tid;
      auto &predictor = predictors[tid];
      if (FLAGS_warmup) {
        PredictionWarmUp(predictor.get(), inputs, &outputs_tid, num_threads,
                         tid);
      }
      PredictionRun(predictor.get(), inputs, &outputs_tid, num_threads, tid);
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

void TestPrediction(const PaddlePredictor::Config *config,
                    const std::vector<std::vector<PaddleTensor>> &inputs,
                    std::vector<std::vector<PaddleTensor>> *outputs,
                    int num_threads, bool use_analysis = FLAGS_use_analysis) {
  PrintConfig(config, use_analysis);
  if (num_threads == 1) {
    TestOneThreadPrediction(config, inputs, outputs, use_analysis);
  } else {
    TestMultiThreadPrediction(config, inputs, outputs, num_threads,
                              use_analysis);
  }
}

void SummarizeAccuracy(float avg_acc1_fp32, float avg_acc1_int8) {
  LOG(INFO) << "--- Accuracy summary --- ";
  LOG(INFO) << "Accepted top1 accuracy drop threshold: "
            << FLAGS_quantized_accuracy
            << ". (condition: (FP32_top1_acc - INT8_top1_acc) <= threshold)";
  LOG(INFO) << "FP32: avg top1 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << avg_acc1_fp32;
  LOG(INFO) << "INT8: avg top1 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << avg_acc1_int8;
}

void SummarizePerformance(float sample_latency_fp32,
                          float sample_latency_int8) {
  // sample latency in ms
  auto throughput_fp32 = 1000.0 / sample_latency_fp32;
  auto throughput_int8 = 1000.0 / sample_latency_int8;
  LOG(INFO) << "--- Performance summary --- ";
  LOG(INFO) << "FP32: avg fps: " << std::fixed << std::setw(6)
            << std::setprecision(4) << throughput_fp32
            << ", avg latency: " << sample_latency_fp32 << " ms";
  LOG(INFO) << "INT8: avg fps: " << std::fixed << std::setw(6)
            << std::setprecision(4) << throughput_int8
            << ", avg latency: " << sample_latency_int8 << " ms";
}

void CompareTopAccuracy(
    const std::vector<std::vector<PaddleTensor>> &output_slots_quant,
    const std::vector<std::vector<PaddleTensor>> &output_slots_ref) {
  if (output_slots_quant.size() == 0 || output_slots_ref.size() == 0)
    throw std::invalid_argument(
        "CompareTopAccuracy: output_slots vector is empty.");

  float total_accs1_quant{0};
  float total_accs1_ref{0};
  for (size_t i = 0; i < output_slots_quant.size(); ++i) {
    PADDLE_ENFORCE(output_slots_quant[i].size() >= 2UL);
    PADDLE_ENFORCE(output_slots_ref[i].size() >= 2UL);
    // second output: acc_top1
    if (output_slots_quant[i][1].lod.size() > 0 ||
        output_slots_ref[i][1].lod.size() > 0)
      throw std::invalid_argument(
          "CompareTopAccuracy: top1 accuracy output has nonempty LoD.");
    if (output_slots_quant[i][1].dtype != paddle::PaddleDType::FLOAT32 ||
        output_slots_ref[i][1].dtype != paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "CompareTopAccuracy: top1 accuracy output is of a wrong type.");
    total_accs1_quant +=
        *static_cast<float *>(output_slots_quant[i][1].data.data());
    total_accs1_ref +=
        *static_cast<float *>(output_slots_ref[i][1].data.data());
  }
  float avg_acc1_quant = total_accs1_quant / output_slots_quant.size();
  float avg_acc1_ref = total_accs1_ref / output_slots_ref.size();

  SummarizeAccuracy(avg_acc1_ref, avg_acc1_quant);
  CHECK_GT(avg_acc1_ref, 0.0);
  CHECK_GT(avg_acc1_quant, 0.0);
  CHECK_LE(avg_acc1_ref - avg_acc1_quant, FLAGS_quantized_accuracy);
}

void CompareDeterministic(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, FLAGS_use_analysis);

  std::vector<PaddleTensor> warmup_outputs, outputs;
  // run num_times to Compare Deterministic Result.
  for (size_t j = 0; j < inputs.size(); j++) {
    // warmup run
    predictor->Run(inputs[j], &warmup_outputs, batch_size);
    for (int i = 0; i < num_times; i++) {
      predictor->Run(inputs[j], &outputs, batch_size);
      CompareResult(outputs, warmup_outputs);
    }
  }
}

void CompareNativeAndAnalysis(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(config, true);
  std::vector<std::vector<PaddleTensor>> native_outputs, analysis_outputs;
  TestOneThreadPrediction(config, inputs, &native_outputs, false);
  TestOneThreadPrediction(config, inputs, &analysis_outputs, true);
  PADDLE_ENFORCE(native_outputs.size() > 0, "Native output is empty.");
  PADDLE_ENFORCE(analysis_outputs.size() > 0, "Analysis output is empty.");
  CompareResult(analysis_outputs.back(), native_outputs.back());
}

void CompareQuantizedAndAnalysis(
    const AnalysisConfig *config, const AnalysisConfig *qconfig,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PADDLE_ENFORCE_EQ(inputs[0][0].shape[0], FLAGS_batch_size,
                    "Input data has to be packed batch by batch.");
  LOG(INFO) << "FP32 & INT8 prediction run: batch_size " << FLAGS_batch_size
            << ", warmup batch size " << FLAGS_warmup_batch_size << ".";

  LOG(INFO) << "--- FP32 prediction start ---";
  auto *cfg = reinterpret_cast<const PaddlePredictor::Config *>(config);
  PrintConfig(cfg, true);
  std::vector<std::vector<PaddleTensor>> analysis_outputs;
  float sample_latency_fp32{-1};
  TestOneThreadPrediction(cfg, inputs, &analysis_outputs, true, VarType::FP32,
                          &sample_latency_fp32);

  LOG(INFO) << "--- INT8 prediction start ---";
  auto *qcfg = reinterpret_cast<const PaddlePredictor::Config *>(qconfig);
  PrintConfig(qcfg, true);
  std::vector<std::vector<PaddleTensor>> quantized_outputs;
  float sample_latency_int8{-1};
  TestOneThreadPrediction(qcfg, inputs, &quantized_outputs, true, VarType::INT8,
                          &sample_latency_int8);

  SummarizePerformance(sample_latency_fp32, sample_latency_int8);
  CompareTopAccuracy(quantized_outputs, analysis_outputs);
}

void CompareNativeAndAnalysis(
    PaddlePredictor *native_pred, PaddlePredictor *analysis_pred,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  int batch_size = FLAGS_batch_size;
  std::vector<PaddleTensor> native_outputs, analysis_outputs;
  native_pred->Run(inputs[0], &native_outputs, batch_size);
  analysis_pred->Run(inputs[0], &analysis_outputs, batch_size);
  CompareResult(analysis_outputs, native_outputs);
}

void CompareAnalysisAndZeroCopy(
    PaddlePredictor::Config *config, PaddlePredictor::Config *config1,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    const std::vector<std::string> &outputs_name) {
  int batch_size = FLAGS_batch_size;
  // analysis
  std::vector<PaddleTensor> analysis_outputs;
  auto predictor = CreateTestPredictor(config, true);
  predictor->Run(inputs[0], &analysis_outputs, batch_size);
  // analysis + zero_copy
  std::vector<ZeroCopyTensor> zerocopy_outputs;
  reinterpret_cast<AnalysisConfig *>(config1)->SwitchUseFeedFetchOps(false);
  predictor = CreateTestPredictor(config1, true);
  ConvertPaddleTensorToZeroCopyTensor(predictor.get(), inputs[0]);
  predictor->ZeroCopyRun();
  for (size_t i = 0; i < outputs_name.size(); i++) {
    ZeroCopyTensor zerocopy_output =
        *predictor->GetOutputTensor(outputs_name[i]).get();
    zerocopy_outputs.emplace_back(zerocopy_output);
    LOG(INFO) << "ZeroCopy output: " << DescribeZeroCopyTensor(zerocopy_output);
  }
  // compare
  CompareResult(analysis_outputs, zerocopy_outputs);
}

void SaveOptimModel(AnalysisConfig *cfg, const std::string &dstPath) {
  auto predictor = CreateTestPredictor(
      reinterpret_cast<const PaddlePredictor::Config *>(cfg),
      FLAGS_use_analysis);
  (static_cast<AnalysisPredictor *>(predictor.get()))->SaveOptimModel(dstPath);
}

template <typename T>
std::string LoDTensorSummary(const framework::LoDTensor &tensor) {
  std::stringstream ss;
  ss << "\n---- tensor ---" << '\n';
  ss << "lod: [";
  for (const auto &level : tensor.lod()) {
    ss << "[ ";
    for (auto i : level) {
      ss << i << ", ";
    }
    ss << "]";
  }
  ss << "]\n";

  ss << "shape: [";
  int size = 1;
  for (int i = 0; i < tensor.dims().size(); i++) {
    int dim = tensor.dims()[i];
    ss << dim << ", ";
    size *= dim;
  }
  ss << "]\n";

  ss << "data: ";
  for (int i = 0; i < std::min(20, size); i++) {
    ss << tensor.data<T>()[i] << " ";
  }
  ss << "\n";

  return ss.str();
}

static bool CompareLoD(const framework::LoD &a, const framework::LoD &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf("lod size not match %d != %d", a.size(),
                                  b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    auto &al = a[i];
    auto &bl = b[i];
    if (al.size() != bl.size()) {
      LOG(ERROR) << string::Sprintf("level size %d != %d", al.size(),
                                    bl.size());
      return false;
    }
  }
  return true;
}

static bool CompareShape(const std::vector<int64_t> &a,
                         const std::vector<int64_t> &b) {
  if (a.size() != b.size()) {
    LOG(ERROR) << string::Sprintf("shape size not match %d != %d", a.size(),
                                  b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      LOG(ERROR) << string::Sprintf("shape %d-th element not match %d != %d", i,
                                    a[i], b[i]);
      return false;
    }
  }
  return true;
}

static bool CompareTensorData(const framework::LoDTensor &a,
                              const framework::LoDTensor &b) {
  auto a_shape = framework::vectorize(a.dims());
  auto b_shape = framework::vectorize(b.dims());
  size_t a_size = std::accumulate(a_shape.begin(), a_shape.end(), size_t{1},
                                  [](int a, int b) { return a * b; });
  size_t b_size = std::accumulate(b_shape.begin(), b_shape.end(), size_t{1},
                                  [](int a, int b) { return a * b; });
  if (a_size != b_size) {
    LOG(ERROR) << string::Sprintf("tensor data size not match, %d != %d",
                                  a_size, b_size);
  }

  for (size_t i = 0; i < a_size; i++) {
    if (a.type() == VarType::FP32) {
      const auto *a_data = a.data<float>();
      const auto *b_data = b.data<float>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f", i, a_data[i],
            b_data[i]);
        return false;
      }
    } else if (a.type() == VarType::INT64) {
      const auto *a_data = a.data<int64_t>();
      const auto *b_data = b.data<int64_t>();
      if (std::abs(a_data[i] - b_data[i]) > 1e-3) {
        LOG(ERROR) << string::Sprintf(
            "tensor data %d-th element not match, %f != %f", i, a_data[i],
            b_data[i]);
        return false;
      }
    }
  }

  return true;
}

static bool CompareTensor(const framework::LoDTensor &a,
                          const framework::LoDTensor &b) {
  if (!CompareLoD(a.lod(), b.lod())) {
    return false;
  }
  if (!CompareShape(framework::vectorize(a.dims()),
                    framework::vectorize(b.dims()))) {
    return false;
  }

  if (!CompareTensorData(a, b)) {
    return false;
  }

  return true;
}

}  // namespace inference
}  // namespace paddle
