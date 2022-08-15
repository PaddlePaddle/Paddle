#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include<ctime>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/infrt/tests/timer.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_bool(tuned_dynamic_shape, false, "use tuned dynamic shape");
DEFINE_bool(tune, false, "tune to get shape range.");

const std::string shape_range_info = "shape_range_info.pbtxt";


std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(100, 0);

  // Open the memory optim.
  config.EnableMemoryOptim();

  int max_batch = 32;
  int max_single_seq_len = 128;
  int opt_single_seq_len = 64;

  std::string input_name0 = "sids";
  std::string input_name1 = "ids";

  std::vector<int> min_shape = {1,1};
  std::vector<int> max_shape = {max_batch,max_single_seq_len};
  std::vector<int> opt_shape = {1,opt_single_seq_len};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {input_name0, min_shape},
      {input_name1, min_shape}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {input_name0, max_shape},
      {input_name1, max_shape}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {input_name0, opt_shape},
      {input_name1, opt_shape}};

  // only kHalf supported
  //config.EnableTensorRtEngine(1 << 30, 1, 5, Config::Precision::kHalf, false,
  //                            false);
  // dynamic shape
  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                opt_input_shape);
 
 config.SwitchIrOptim(true);
 config.SwitchIrDebug();
// config.pass_builder()->DeletePass("multihead_matmul_fuse_pass_v2");
// config.pass_builder()->DeletePass("skip_layernorm_fuse_pass");
// config.pass_builder()->DeletePass("embedding_eltwise_layernorm_fuse_pass");

  if (FLAGS_tuned_dynamic_shape) {
    config.EnableTunedTensorRtDynamicShape(shape_range_info, true);
  }
  if (FLAGS_tune) {
    config.CollectShapeRangeInfo(shape_range_info);
  }

  LOG(INFO) << config.Summary();


 return CreatePredictor(config);
}

void run(Predictor *predictor, std::vector<float> *out_data) {
  const int run_batch = 10;
  const int run_seq_len = 128;
  const int max_seq_len = 128;

  int64_t i1[run_seq_len*run_batch] = {1,6100,2,6100,703,136,102,1882,729,2};
  int64_t i2[run_seq_len*run_batch] = {0,0,0,1,1,1,1,1,1,1};

  auto input_names = predictor->GetInputNames();
  // first input
  auto input_t1 = predictor->GetInputHandle(input_names[0]);
  input_t1->Reshape({run_batch, run_seq_len});
  input_t1->CopyFromCpu(i1);

  // second input
  auto input_t2 = predictor->GetInputHandle(input_names[1]);
  input_t2->Reshape({run_batch, run_seq_len});
  input_t2->CopyFromCpu(i2);


  clock_t startTime,endTime;
  for(int i = 0; i<1; i++){
    CHECK(predictor->Run());
  }

  infrt::tests::BenchmarkStats benchmark;
  for(int i = 0; i<100; i++){
    benchmark.Start();
    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
    benchmark.Stop();
  }
  std::cout << benchmark.Summerize({0, 0.5, 1.0}) << '\n';
  
  //std::cout << "The run time is: " <<(double)(endTime - startTime)*0.1 / CLOCKS_PER_SEC << "s" << std::endl;
  return;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  std::vector<float> out_data;
  run(predictor.get(), &out_data);

  for (auto r : out_data) {
//    LOG(INFO) << r;
  }
  return 0;
}
