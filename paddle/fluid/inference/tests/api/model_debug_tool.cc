// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tests/api/feed_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/infrt/tests/timer.h"

using paddle::CreatePaddlePredictor;
using paddle::PaddlePredictor;
using paddle_infer::Config;
using paddle_infer::DataType;

DEFINE_string(model_name, "", "");
DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "");
// DEFINE_bool(tuned_dynamic_shape, false, "use tuned dynamic shape");
// DEFINE_bool(tune, false, "tune to get shape range.");
DEFINE_bool(enable_cinn, false, "enable cinn");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(model_path_prefix, "", "The prefix of model path.");
DEFINE_string(inputs_path_prefix, "", "The prefix of input files path.");

LoadConfig ae_freshquery_config() {
  LoadConfig config;
  FLAGS_model_dir =
      FLAGS_model_path_prefix +
      "/data_B327A5DB886EDC7260A5AAAD4E9EEC60/"
      "ae_freshquery_classify_hutianyi03_20220608-12/inference_model/";
  config.file_path =
      FLAGS_inputs_path_prefix +
      "/ae_freshquery_classify_hutianyi03_20220608/paddle_debug.txt";
  config.capability = 100;
  config.ids.resize(4);
  config.ids[0] = {"read_file_0.tmp_0", "input_id", DataType::INT64};
  config.ids[1] = {"read_file_0.tmp_1", "loc_id", DataType::INT64};
  config.ids[2] = {"read_file_0.tmp_2", "sent_id", DataType::INT64};
  config.ids[3] = {"read_file_0.tmp_3", "mask", DataType::FLOAT32};
  return config;
}

LoadConfig quality_ernie_yangboyu01_config() {
  LoadConfig config;
  FLAGS_model_dir =
      FLAGS_model_path_prefix +
      "/data_AB021FD1AD84BFCF73346A1C05745B80/"
      "quality_ernie_yangboyu01_20220418-6/qq_recall_v3_0_inference_model/";
  config.file_path = FLAGS_inputs_path_prefix +
                     "/quality_ernie_yangboyu01_20220418/paddle_debug.txt";
  config.capability = 100;
  config.ids.resize(4);
  config.ids[0] = {"read_file_0.tmp_5", "input_id", DataType::INT64};
  config.ids[1] = {"read_file_0.tmp_7", "loc_id", DataType::INT64};
  config.ids[2] = {"read_file_0.tmp_6", "sent_id", DataType::INT64};
  config.ids[3] = {"read_file_0.tmp_9", "mask", DataType::FLOAT32};
  config.others = {
      {"read_file_0.tmp_8", "input_4", DataType::INT64},
  };
  return config;
}

LoadConfig ann_ernie_vip_query_zhangzhe20_config() {
  LoadConfig config;
  FLAGS_model_file =
      FLAGS_model_path_prefix +
      "/data_8E0DEDE978D02DE539A333928BD8C363/"
      "ann_ernie_vip_query_zhangzhe20_20221109_base_zdf_210623_query-2/"
      "inference_model/__model__";
  FLAGS_params_file =
      FLAGS_model_path_prefix +
      "/data_8E0DEDE978D02DE539A333928BD8C363/"
      "ann_ernie_vip_query_zhangzhe20_20221109_base_zdf_210623_query-2/"
      "inference_model/__params__";
  config.file_path =
      FLAGS_inputs_path_prefix +
      "/ann_ernie_vip_query_zhangzhe20_20221109_base_zdf_210623_query/"
      "paddle_debug.txt";
  config.capability = 100;
  config.ids.resize(4);
  config.ids[0] = {"read_file_0.tmp_0", "input_id", DataType::INT64};
  config.ids[1] = {"read_file_0.tmp_1", "sent_id", DataType::INT64};
  config.ids[2] = {"read_file_0.tmp_2", "loc_id", DataType::INT64};
  config.ids[3] = {"read_file_0.tmp_4", "mask", DataType::FLOAT32};
  return config;
}

LoadConfig ann_ernie_vip_query_config_zhangjiaming04() {
  LoadConfig config;
  FLAGS_model_file =
      FLAGS_model_path_prefix +
      "/data_201EB818283DAFF7B98C98870EF145E7/"
      "ann_ernie_vip_query_zhangjiaming04_20221114_cq_v8_query-1/"
      "model_600_query/__model__";
  FLAGS_params_file =
      FLAGS_model_path_prefix +
      "/data_201EB818283DAFF7B98C98870EF145E7/"
      "ann_ernie_vip_query_zhangjiaming04_20221114_cq_v8_query-1/"
      "model_600_query/__params__";
  config.file_path = FLAGS_inputs_path_prefix +
                     "/ann_ernie_vip_query_zhangjiaming04_20221114_cq_v8_query/"
                     "paddle_debug.txt";
  config.capability = 100;
  config.ids.resize(4);
  config.ids[0] = {"qb", "input_id", DataType::INT64};
  config.ids[1] = {"cast_1.tmp_0", "loc_id", DataType::INT64};
  config.ids[2] = {"qb_type", "sent_id", DataType::INT64};
  config.ids[3] = {"cast_0.tmp_0", "mask", DataType::FLOAT32};
  return config;
}

LoadConfig staratlas_demand_classify() {
  LoadConfig config;
  FLAGS_model_file =
      FLAGS_model_path_prefix +
      "/data_051D1CE2EDEB76B3D902C4095EB05022/"
      "staratlas_demand_classify_xijian_20220104_new_system_sample-3/"
      "infer_model/model";
  FLAGS_params_file =
      FLAGS_model_path_prefix +
      "/data_051D1CE2EDEB76B3D902C4095EB05022/"
      "staratlas_demand_classify_xijian_20220104_new_system_sample-3/"
      "infer_model/params";
  config.file_path =
      FLAGS_inputs_path_prefix +
      "/staratlas_demand_classify_xijian_20220104_new_system_sample/"
      "paddle_debug.txt";
  config.capability = 100;
  config.others = {
      {"read_file_0.tmp_0", "input_0", DataType::INT64},
      {"read_file_0.tmp_2", "input_1", DataType::INT64},
      {"read_file_0.tmp_1", "input_2", DataType::INT64},
      {"read_file_0.tmp_3", "input_3", DataType::FLOAT32},
  };
  return config;
}

std::shared_ptr<PaddlePredictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  } else {
    config.SetModel(FLAGS_model_file, FLAGS_params_file);
  }
  config.EnableUseGpu(100, 0, paddle::AnalysisConfig::Precision::kHalf);
  config.EnableMemoryOptim();

  config.SwitchIrOptim(true);
  config.SwitchIrDebug();
  auto pass_builder = config.pass_builder();
  pass_builder->DeletePass("constant_folding_pass");

  if (FLAGS_enable_cinn) {
    config.Exp_EnableCINNCompiler();
    config.EnableMemoryOptim(false);
    config.pass_builder()->ClearPasses();
    config.pass_builder()->AppendPass("auto_mixed_precision_pass");
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_v2_to_mul_pass");
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_v2_to_matmul_pass");
    config.pass_builder()->AppendPass("gpu_cpu_map_matmul_to_mul_pass");
    config.pass_builder()->AppendPass("build_cinn_pass");
  }

  LOG(INFO) << "Used passes: " << config.pass_builder()->DebugString();
  LOG(INFO) << config.Summary();

  return CreatePaddlePredictor(config);
}

void run(PaddlePredictor* predictor, const LoadConfig& config) {
  RuntimeContext ctx{std::move(config)};
  FeedData data;
  auto ret = std::async(std::launch::async, file_loader, std::ref(ctx), &data);
  while (true) {
    auto inputs = data.pop();
    if (inputs.empty()) {
      break;
    }
    std::vector<paddle::PaddleTensor> feed_tensors;
    for (auto& input : inputs) {
      feed_tensors.emplace_back(std::move(input.release()));
    }
    std::vector<paddle::PaddleTensor> fetch_tensors;
    infrt::tests::BenchmarkStats benchmark;
    predictor->Run(feed_tensors, &fetch_tensors);
    benchmark.Start();
    predictor->Run(std::move(feed_tensors), &fetch_tensors);
    benchmark.Stop();
    LOG(INFO) << benchmark.Summerize({0, 0.5, 0.8, 0.9, 0.95, 0.99});
    for (auto& fetch : fetch_tensors) {
      // print_tensor(fetch);
    }
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  // paddle::framework::InitGflags(
  //     {"--tryfromenv=allow_cinn_ops,deny_cinn_ops,enable_pe_launch_cinn"});

  LOG(INFO) << "FLAGS_repeats: " << FLAGS_repeats;
  LoadConfig config;
  if (FLAGS_model_name == "ae_freshquery_classify_hutianyi03_20220608-12") {
    config = ae_freshquery_config();
  } else if (FLAGS_model_name == "quality_ernie_yangboyu01_20220418-6") {
    config = quality_ernie_yangboyu01_config();
  } else if (FLAGS_model_name ==
             "ann_ernie_vip_query_zhangzhe20_20221109_base_zdf_210623_query-"
             "2") {
    config = ann_ernie_vip_query_zhangzhe20_config();
  } else if (FLAGS_model_name ==
             "ann_ernie_vip_query_zhangjiaming04_20221114_cq_v8_query-1") {
    config = ann_ernie_vip_query_config_zhangjiaming04();
  } else if (FLAGS_model_name == "staratlas_demand_classify") {
    config = staratlas_demand_classify();
  } else {
    LOG(FATAL) << "illegal model name: " << FLAGS_model_name;
  }
  auto predictor = InitPredictor();
  std::vector<float> out_data;
  run(predictor.get(), std::move(config));
  return 0;
}
