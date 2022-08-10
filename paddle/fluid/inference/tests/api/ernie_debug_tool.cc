// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/init.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(input_file, "", "Input data file of model");
DEFINE_int32(max_sequence_length,
             128,
             "Max sequence length of the inference model.");
DEFINE_string(
    mode,
    "no_opt",
    "Inference mode. choice = ['no_opt', 'trt', 'trt_varlen', 'all']");
DEFINE_bool(use_int8, false, "is open int8");
DEFINE_int32(warmup_times, 10, "warmup times");
DEFINE_int32(repeats, 100, "repeate times");
DEFINE_int32(run_batch, 1, "run batch");
DEFINE_int32(max_batch, 20, "max batch");
DEFINE_int32(min_subgraph_size, 5, "min subgraph size");
DEFINE_bool(req_with_batch, false, "input with batch");
DEFINE_int32(use_build_trt_engine_info, 1, "use_build_trt_engine_info");
DEFINE_bool(paddle_fp16, false, "paddle phi fp16");
DEFINE_bool(clear_passes, false, "clear all passes");
DEFINE_bool(delete_pass, false, "delete pass");
DEFINE_bool(enable_cinn, false, "enable cinn");

DECLARE_string(allow_cinn_ops);
DECLARE_string(deny_cinn_ops);

constexpr int output_cut_off = 10;
int max_single_seq_len = FLAGS_max_sequence_length;
std::string input_data_type = "non-varlen";

class Timer {
  // Timer, count in ms
 public:
  Timer() { reset(); }
  void start() { start_t = std::chrono::high_resolution_clock::now(); }
  void stop() {
    auto end_t = std::chrono::high_resolution_clock::now();
    typedef std::chrono::microseconds ms;
    auto diff = end_t - start_t;
    ms counter = std::chrono::duration_cast<ms>(diff);
    total_time += counter.count();
  }
  void reset() { total_time = 0.; }
  double report() { return total_time / 1000.0; }

 private:
  double total_time;
  std::chrono::high_resolution_clock::time_point start_t;
};

struct ernie_data {
  std::string name;
  std::string data_type = "int32_t";
  std::vector<int> data;
  std::vector<int> varlen_data;  // used for running varlen
  std::vector<float> float_data;
  std::vector<int64_t> int64_data;
  std::vector<int> shape;
};
std::map<std::string, ernie_data> file_input;

void verify_input_name(const std::vector<std::string> &inputnames) {
  if (inputnames[3] != file_input["mask"].name) {
    std::cerr << "slot 3 should be mask\n";
    return;
  }
}

void process_ext_input(const std::vector<std::string> &inputnames,
                       Predictor *predictor) {
  // 0-3 are necessary inputs
  for (int i = 4; i < inputnames.size(); i++) {
    for (auto const it : file_input) {
      auto const &er_data = it.second;
      if (er_data.name == inputnames[i]) {
        // auto const &er_data = file_input[inputnames[i]];
        auto input = predictor->GetInputHandle(er_data.name);
        input->Reshape(er_data.shape);
        if (er_data.data_type == "int32_t") {
          std::vector<int> inum;
          inum.insert(inum.end(), er_data.data.begin(), er_data.data.end());
          input->CopyFromCpu(inum.data());
        } else if (er_data.data_type == "float") {
          std::vector<float> inum;
          inum.insert(
              inum.end(), er_data.float_data.begin(), er_data.float_data.end());
          input->CopyFromCpu(inum.data());
        } else if (er_data.data_type == "int64_t") {
          std::vector<int64_t> inum;
          inum.insert(
              inum.end(), er_data.int64_data.begin(), er_data.int64_data.end());
          input->CopyFromCpu(inum.data());
        }
        break;
      }
    }
  }
}

std::shared_ptr<Predictor> init_pred_no_varlen() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(100, 0);

  // Open the memory optim.
  config.EnableMemoryOptim();
  if (FLAGS_mode == "trt") {
    int opt_single_seq_len = max_single_seq_len / 2;

    std::string input_name0 = file_input["input_id"].name;
    std::string input_name1 = file_input["sent_id"].name;
    std::string input_name2 = file_input["loc_id"].name;
    std::string input_name3 = file_input["mask"].name;

    std::vector<int> min_shape = {1, 1, 1};
    std::vector<int> max_shape = {FLAGS_max_batch, max_single_seq_len, 1};
    std::vector<int> opt_shape = {1, opt_single_seq_len, 1};
    // Set the input's min, max, opt shape
    std::map<std::string, std::vector<int>> min_input_shape = {
        {input_name0, min_shape},
        {input_name1, min_shape},
        {input_name2, min_shape},
        {input_name3, min_shape}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {input_name0, max_shape},
        {input_name1, max_shape},
        {input_name2, max_shape},
        {input_name3, max_shape}};
    std::map<std::string, std::vector<int>> opt_input_shape = {
        {input_name0, opt_shape},
        {input_name1, opt_shape},
        {input_name2, opt_shape},
        {input_name3, opt_shape}};

    // only kHalf supported
    config.EnableTensorRtEngine(1 << 30,
                                1,
                                FLAGS_min_subgraph_size,
                                Config::Precision::kHalf,
                                false,
                                false);
    // dynamic shape
    config.SetTRTDynamicShapeInfo(
        min_input_shape, max_input_shape, opt_input_shape);
  }

  config.SwitchIrOptim(1);

  // if (FLAGS_paddle_fp16) {
  //   config.Exp_EnableUseGpuFp16();
  // }

  if (FLAGS_delete_pass) {
    config.pass_builder()->DeletePass("embedding_eltwise_layernorm_fuse_pass");
  }

  if (FLAGS_enable_cinn) {
    config.EnableMemoryOptim(false);
    config.pass_builder()->ClearPasses();
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_v2_to_mul_pass");
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_v2_to_matmul_pass");
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_to_mul_pass");
    config.pass_builder()->AppendPass("build_cinn_pass");
    config.pass_builder()->AppendPass("graph_viz_pass");
  }

  if (FLAGS_clear_passes) {
    config.pass_builder()->ClearPasses();
  }

  LOG(INFO) << config.pass_builder()->DebugString();

  return CreatePredictor(config);
}

void run_no_varlen(Predictor *predictor,
                   std::vector<float> *out_data,
                   double &qps) {
  std::vector<int64_t> i1;
  std::vector<int64_t> i2;
  std::vector<int64_t> i3;
  std::vector<float> i4;
  int run_seq_len = 0;
  if (FLAGS_req_with_batch) {
    run_seq_len =
        static_cast<int>(file_input["input_id"].data.size()) / FLAGS_run_batch;
  } else {
    run_seq_len = static_cast<int>(file_input["input_id"].data.size());
  }

  for (int i = 0; i < FLAGS_run_batch; i++) {
    i1.insert(i1.end(),
              file_input["input_id"].data.begin(),
              file_input["input_id"].data.end());
    i2.insert(i2.end(),
              file_input["sent_id"].data.begin(),
              file_input["sent_id"].data.end());
    i3.insert(i3.end(),
              file_input["loc_id"].data.begin(),
              file_input["loc_id"].data.end());
    i4.insert(i4.end(),
              file_input["mask"].data.begin(),
              file_input["mask"].data.end());
    if (FLAGS_req_with_batch)
      break;  // insert once if req already has batched datas
  }

  auto input_names = predictor->GetInputNames();
  verify_input_name(input_names);

  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    // first input
    auto input_t1 = predictor->GetInputHandle(file_input["input_id"].name);
    input_t1->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t1->CopyFromCpu(i1.data());

    // second input
    auto input_t2 = predictor->GetInputHandle(file_input["sent_id"].name);
    input_t2->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t2->CopyFromCpu(i2.data());

    // third input
    auto input_t3 = predictor->GetInputHandle(file_input["loc_id"].name);
    input_t3->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t3->CopyFromCpu(i3.data());

    // fourth input
    auto input_t4 = predictor->GetInputHandle(file_input["mask"].name);
    input_t4->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t4->CopyFromCpu(i4.data());

    process_ext_input(input_names, predictor);

    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }

  Timer pred_timer;
  pred_timer.start();  // start timer
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    // first input
    auto input_t1 = predictor->GetInputHandle(file_input["input_id"].name);
    input_t1->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t1->CopyFromCpu(i1.data());

    // second input
    auto input_t2 = predictor->GetInputHandle(file_input["sent_id"].name);
    input_t2->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t2->CopyFromCpu(i2.data());

    // third input
    auto input_t3 = predictor->GetInputHandle(file_input["loc_id"].name);
    input_t3->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t3->CopyFromCpu(i3.data());

    // fourth input
    auto input_t4 = predictor->GetInputHandle(file_input["mask"].name);
    input_t4->Reshape({FLAGS_run_batch, run_seq_len, 1});
    input_t4->CopyFromCpu(i4.data());

    process_ext_input(input_names, predictor);

    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  pred_timer.stop();  // stop timer
  if (FLAGS_req_with_batch) {
    LOG(INFO) << "The cost time: " << pred_timer.report() / FLAGS_repeats
              << " ms";
    qps = FLAGS_repeats / (pred_timer.report() / 1000.0);
  } else {
    qps = FLAGS_repeats * FLAGS_run_batch / (pred_timer.report() / 1000.0);
  }
  return;
}

std::shared_ptr<Predictor> init_pred_var_len() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(100, 0);

  // Open the memory optim.
  config.EnableMemoryOptim();

  int opt_single_seq_len = max_single_seq_len / 2;
  int min_batch_seq_len = 1;
  int max_batch_seq_len = max_single_seq_len * FLAGS_max_batch;
  int opt_batch_seq_len = max_batch_seq_len / 2;

  std::string input_name0 = file_input["input_id"].name;
  std::string input_name1 = file_input["sent_id"].name;
  std::string input_name2 = file_input["loc_id"].name;
  std::string input_name3 = file_input["mask"].name;

  std::vector<int> min_shape = {min_batch_seq_len};
  std::vector<int> max_shape = {max_batch_seq_len};
  std::vector<int> opt_shape = {opt_batch_seq_len};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {input_name0, min_shape},
      {input_name1, min_shape},
      {input_name2, {1}},
      {input_name3, {1, 1, 1}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {input_name0, max_shape},
      {input_name1, max_shape},
      {input_name2, {FLAGS_max_batch + 1}},
      {input_name3, {1, max_single_seq_len, 1}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {input_name0, opt_shape},
      {input_name1, opt_shape},
      {input_name2, {FLAGS_max_batch + 1}},
      {input_name3, {1, opt_single_seq_len, 1}}};

  // only kHalf supported
  config.EnableTensorRtEngine(1 << 30,
                              1,
                              FLAGS_min_subgraph_size,
                              Config::Precision::kHalf,
                              false,
                              false);
  // erinie varlen must be used with dynamic shape
  config.SetTRTDynamicShapeInfo(
      min_input_shape, max_input_shape, opt_input_shape);
  // erinie varlen must be used with oss
  // config.EnableTensorRtOSS();

  // int8 config
  if (FLAGS_use_int8) {
    config.EnableTensorRtEngine(1 << 30,
                                FLAGS_max_batch,
                                FLAGS_min_subgraph_size,
                                paddle::AnalysisConfig::Precision::kInt8,
                                FLAGS_use_build_trt_engine_info,
                                false);
  }

  return CreatePredictor(config);
}

void run_var_len(Predictor *predictor,
                 std::vector<float> *out_data,
                 double &qps) {
  //  const int run_seq_len = 71;
  //  const int max_seq_len = 128;
  int run_seq_len = file_input["input_id"].varlen_data.size();
  std::vector<int32_t> i1;
  std::vector<int32_t> i2;

  for (int i = 0; i < FLAGS_run_batch; i++) {
    i1.insert(i1.end(),
              file_input["input_id"].varlen_data.begin(),
              file_input["input_id"].varlen_data.end());
    i2.insert(i2.end(),
              file_input["sent_id"].varlen_data.begin(),
              file_input["sent_id"].varlen_data.end());
    if (FLAGS_req_with_batch)
      break;  // insert once if req already has batched datas
  }

  // shape info of this batch
  std::vector<int32_t> i3(file_input["loc_id"].varlen_data.begin(),
                          file_input["loc_id"].varlen_data.end());
  // max_single_seq_len represents the max sentence length of all the sentences,
  // only length of input i4 is useful, data means nothing.
  std::vector<int32_t> i4(run_seq_len, 0);

  auto input_names = predictor->GetInputNames();
  verify_input_name(input_names);

  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    // first input
    auto input_t1 = predictor->GetInputHandle(file_input["input_id"].name);
    input_t1->Reshape({run_seq_len});
    input_t1->CopyFromCpu(i1.data());

    // second input
    auto input_t2 = predictor->GetInputHandle(file_input["sent_id"].name);
    input_t2->Reshape({run_seq_len});
    input_t2->CopyFromCpu(i2.data());

    // third input
    auto input_t3 = predictor->GetInputHandle(file_input["loc_id"].name);
    input_t3->Reshape({FLAGS_run_batch + 1});
    input_t3->CopyFromCpu(i3.data());

    // fourth input
    auto input_t4 = predictor->GetInputHandle(file_input["mask"].name);
    input_t4->Reshape({1, max_single_seq_len, 1});
    input_t4->CopyFromCpu(i4.data());

    process_ext_input(input_names, predictor);

    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }

  Timer pred_timer;
  pred_timer.start();  // start timer
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    // first input
    auto input_t1 = predictor->GetInputHandle(file_input["input_id"].name);
    input_t1->Reshape({run_seq_len});
    input_t1->CopyFromCpu(i1.data());

    // second input
    auto input_t2 = predictor->GetInputHandle(file_input["sent_id"].name);
    input_t2->Reshape({run_seq_len});
    input_t2->CopyFromCpu(i2.data());

    // third input
    auto input_t3 = predictor->GetInputHandle(file_input["loc_id"].name);
    input_t3->Reshape({FLAGS_run_batch + 1});
    input_t3->CopyFromCpu(i3.data());

    // fourth input
    auto input_t4 = predictor->GetInputHandle(file_input["mask"].name);
    input_t4->Reshape({1, max_single_seq_len, 1});
    input_t4->CopyFromCpu(i4.data());

    process_ext_input(input_names, predictor);

    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  pred_timer.stop();  // stop timer
  if (FLAGS_req_with_batch) {
    qps = FLAGS_repeats / (pred_timer.report() / 1000.0);
  } else {
    qps = FLAGS_repeats * FLAGS_run_batch / (pred_timer.report() / 1000.0);
  }

  return;
}

void read_file() {
  std::ifstream file(FLAGS_input_file);
  if (!file) std::cerr << FLAGS_input_file << " doesn't exist" << std::endl;
  std::string line;
  while (file.good()) {
    std::getline(file, line);
    size_t pos = line.find(':');
    std::string in(line.substr(0, pos));
    size_t underline = in.find_last_of('_');
    std::string type = in.substr(underline + 1);
    std::string loc_name(in.substr(0, underline));
    if (type == "data") {
      std::istringstream numstr(line.substr(pos + 1));
      int num = 0;
      char comma = 0;
      int64_t num64 = 0;
      float fp_num = 0;
      std::string data_type = file_input[loc_name].data_type;
      while (numstr.good()) {
        if (data_type == "int32_t")
          numstr >> num;
        else if (data_type == "float")
          numstr >> fp_num;
        else if (data_type == "int64_t")
          numstr >> num64;
        if (!numstr.good()) break;
        numstr >> comma;
        if (data_type == "int32_t")
          file_input[loc_name].data.push_back(num);
        else if (data_type == "float")
          file_input[loc_name].float_data.push_back(fp_num);
        else if (data_type == "int64_t")
          file_input[loc_name].int64_data.push_back(num64);
      }
    } else if (type == "name") {
      file_input[loc_name].name = line.substr(pos + 1, line.length());
    } else if (type == "type") {
      file_input[loc_name].data_type = line.substr(pos + 1);
      if (file_input[loc_name].data_type != "float" &&
          file_input[loc_name].data_type != "int32_t" &&
          file_input[loc_name].data_type != "int64_t") {
        std::cerr
            << "unknown data type, only float32, int32, int64 is supported.\n";
      }
    } else if (type == "shape") {
      std::istringstream numstr(line.substr(pos + 1));
      int num = 0;
      char comma = 0;
      while (numstr.good()) {
        numstr >> num;
        if (!numstr.good()) break;
        numstr >> comma;
        file_input[loc_name].shape.push_back(num);
      }
      if (file_input[loc_name].shape[0] != 1) {
        std::cerr << "please input 1 batch data\n";
      }

      // modify num on batch size
      file_input[loc_name].shape[0] = FLAGS_run_batch;
    } else {
      std::cerr << "unkown type!\n";
    }
  }
  if (file_input["input_id"].data.size() >
          max_single_seq_len * FLAGS_run_batch ||
      file_input["input_id"].data.size() <= 0) {
    std::cerr << "Input data size should < max_single_seq_len * run_batch  and "
                 ">= 0.\n";
    return;
  }
  if (file_input["input_id"].data.size() != file_input["loc_id"].data.size()) {
    input_data_type = "var-len";
    return;
  }
}

void generate_varlen_data() {
  int run_seq_len = 0;
  if (FLAGS_req_with_batch) {
    run_seq_len =
        static_cast<int>(file_input["input_id"].data.size()) / FLAGS_run_batch;
  } else {
    run_seq_len = static_cast<int>(file_input["input_id"].data.size());
  }

  int total_seq_len = 0;
  std::vector<int32_t> loc;
  loc.emplace_back(0);
  for (int i = 0; i < FLAGS_run_batch; i++) {
    int seq_len = 0;
    for (int j = 0; j < run_seq_len; j++) {
      int32_t input_id = file_input["input_id"].data[i * run_seq_len + j];
      int32_t sent_id = file_input["sent_id"].data[i * run_seq_len + j];
      if (input_id == 0) break;  // if is padding
      file_input["input_id"].varlen_data.emplace_back(input_id);
      file_input["sent_id"].varlen_data.emplace_back(sent_id);
      ++seq_len;
    }
    total_seq_len += seq_len;
    loc.emplace_back(total_seq_len);
  }
  file_input["loc_id"].varlen_data = loc;
}

void generate_non_varlen_data() {
  auto &input_ids = file_input["input_id"];
  auto &sent_ids = file_input["sent_id"];
  auto &loc_ids = file_input["loc_id"];
  auto &mask_ids = file_input["mask"];

  // 1. save origin var-len input into ernie_data.varlen_data
  input_ids.varlen_data = input_ids.data;
  sent_ids.varlen_data = sent_ids.data;
  loc_ids.varlen_data = loc_ids.data;
  mask_ids.varlen_data = mask_ids.data;

  // 2. clear ernie_data.data
  input_ids.data.clear();
  sent_ids.data.clear();
  loc_ids.data.clear();
  mask_ids.data.clear();

  // 3. convert var-len input into non-varlen, saved in ernie_data.data
  int max_seq_len = 0, pos = 0;
  const auto &loc = loc_ids.varlen_data;
  for (int i = 1; i < loc.size(); i++) {
    if (loc[i] - loc[i - 1] > max_seq_len) max_seq_len = loc[i] - loc[i - 1];
  }
  for (int i = 0; i < FLAGS_run_batch; i++) {
    for (int j = 0; j < max_seq_len; j++) {
      if (j < loc[i + 1] - loc[i]) {
        input_ids.data.emplace_back(input_ids.varlen_data[pos]);
        sent_ids.data.emplace_back(sent_ids.varlen_data[pos]);
        loc_ids.data.emplace_back(j);
        mask_ids.data.emplace_back(1);
        ++pos;
      } else {
        input_ids.data.emplace_back(0);
        sent_ids.data.emplace_back(0);
        loc_ids.data.emplace_back(0);
        mask_ids.data.emplace_back(0);
      }
    }
  }
}

void complete_ernie_data() {
  if (input_data_type == "non-varlen") {
    if (FLAGS_mode == "all" or FLAGS_mode == "trt_varlen") {
      generate_varlen_data();
    }
    return;
  }
  if (input_data_type == "var-len") {
    if (FLAGS_mode == "all" or FLAGS_mode != "trt_varlen") {
      generate_non_varlen_data();
    } else {
      file_input["input_id"].varlen_data = file_input["input_id"].data;
      file_input["sent_id"].varlen_data = file_input["sent_id"].data;
      file_input["loc_id"].varlen_data = file_input["loc_id"].data;
      file_input["mask"].varlen_data = file_input["mask"].data;
    }
    return;
  }
}

void run_model(const std::string &mode,
               std::vector<float> &out_data,
               double &qps) {
  std::shared_ptr<Predictor> predictor;
  if (mode == "trt" || mode == "no_opt") {
    predictor = init_pred_no_varlen();
    run_no_varlen(predictor.get(), &out_data, qps);
  } else if (mode == "trt_varlen") {
    predictor = init_pred_var_len();
    run_var_len(predictor.get(), &out_data, qps);
  } else {
    std::cerr << mode << " is unsuppprted\n";
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::framework::InitGflags({"--tryfromenv=allow_cinn_ops,deny_cinn_ops,enable_pe_launch_cinn"});

  std::vector<std::string> run_modes;
  std::vector<double> qps_res;
  std::vector<std::vector<float>> out_datas;

  read_file();
  complete_ernie_data();

  if (FLAGS_mode == "all")
    run_modes = {"no_opt", "trt", "trt_varlen"};
  else
    run_modes = {FLAGS_mode};
  out_datas.resize(run_modes.size());
  qps_res.resize(run_modes.size());
  for (int i = 0; i < run_modes.size(); i++) {
    run_model(run_modes[i], out_datas[i], qps_res[i]);
  }

  for (int k = 0; k < run_modes.size(); k++) {
    int len = out_datas[k].size() / FLAGS_run_batch;
    int out_num = len > output_cut_off ? output_cut_off : len;
    std::cout << run_modes[k] << " output:\n";
    for (int i = 0; i < FLAGS_run_batch; i++) {
      std::cout << "output" << i << ":";
      for (int j = 0; j < out_num; j++) {
        std::cout << out_datas[k][len * i + j] << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "QPS : " << qps_res[k] << std::endl;
  }

  return 0;
}
