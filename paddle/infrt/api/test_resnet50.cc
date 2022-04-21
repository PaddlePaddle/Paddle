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
#include <gperftools/heap-profiler.h>
#include <chrono>
#include "paddle/infrt/api/infrt_api.h"
#include "paddle/infrt/backends/host/phi_allocator.h"

using infrt::InfRtConfig;
using infrt::InfRtPredictor;
using infrt::CreateInfRtPredictor;

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); }
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

int parseLine(char* line) {
  // This assumes that a digit will be found and the line ends in " Kb".
  int i = strlen(line);
  const char* p = line;
  while (*p < '0' || *p > '9') p++;
  line[i - 3] = '\0';
  i = atoi(p);
  return i;
}

int getValue() {  // Note: this value is in KB!
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      result = parseLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}

DEFINE_string(model_dir, "", "model dir");
DEFINE_string(params_dir, "", "params dir");
DEFINE_int32(warmup, 0, "warmups");
DEFINE_int32(repeats, 1, "repetas");
DEFINE_bool(enable_profile, false, "profile");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_enable_profile) HeapProfilerStart("start_profile");

  InfRtConfig config;
  config.enable_tensorrt();
  // config.set_model_dir("/wilber/repo/Paddle/build/models/resnet50/model.pdmodel");
  // config.set_param_dir("/wilber/repo/Paddle/build/models/resnet50/model.pdiparams");
  // config.set_model_dir("/wilber/repo/Paddle/build/models/clip/model.pdmodel");
  // config.set_param_dir("/wilber/repo/Paddle/build/models/clip/model.pdiparams");
  config.set_model_dir(FLAGS_model_dir);
  config.set_param_dir(FLAGS_params_dir);

  int batch_size = 1;
  std::vector<int> input_shape = {batch_size, 3, 256, 256};
  std::vector<float> input_data(batch_size * 3 * 256 * 256);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = i % 255 * 0.1;
  // std::vector<float> out_data;

  std::unique_ptr<InfRtPredictor> predictor = CreateInfRtPredictor(config);
  if (FLAGS_enable_profile) HeapProfilerDump("After CreateInfRtPredictor");

  ::phi::DenseTensor* input = predictor->GetInput(0);
  // auto* output = predictor->GetOutput(0);

  ::infrt::backends::CpuPhiAllocator cpu_allocator;
  input->Resize({1, 3, 256, 256});
  input->AllocateFrom(&cpu_allocator, ::phi::DataType::FLOAT32);
  auto* in_data = reinterpret_cast<float*>(input->data());
  for (size_t i = 0; i < input_data.size(); ++i) in_data[i] = input_data[i];

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto st = time();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor->Run();
  }
  if (FLAGS_enable_profile) HeapProfilerDump("After run");
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
  if (FLAGS_enable_profile) HeapProfilerStop();
  return 0;
}
