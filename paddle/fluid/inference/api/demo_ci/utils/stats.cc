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

#include "stats.h"
#include <iostream>
#include <sstream>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {

namespace {

template <typename T>
void SkipFirstNData(std::vector<T>& v, int n) {
  std::vector<T>(v.begin() + n, v.end()).swap(v);
}

template <typename T>
T FindAverage(const std::vector<T>& v) {
  if (v.size() == 0)
    throw std::invalid_argument("FindAverage: vector is empty.");
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
T FindPercentile(std::vector<T> v, int p) {
  if (v.size() == 0)
    throw std::invalid_argument("FindPercentile: vector is empty.");
  std::sort(v.begin(), v.end());
  if (p == 100) return v.back();
  int i = v.size() * p / 100;
  return v[i];
}

template <typename T>
T FindStandardDev(std::vector<T> v) {
  if (v.size() == 0)
    throw std::invalid_argument("FindStandardDev: vector is empty.");
  T mean = FindAverage(v);
  T var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i] - mean) * (v[i] - mean);
  }
  var /= v.size();
  T std = sqrt(var);
  return std;
}

}  // namespace

// gather statistics from the output
void Stats::Gather(const std::vector<PaddleTensor>& output_slots,
                   double batch_time,
                   int iter) {
  latencies.push_back(batch_time);
  double fps = batch_size / batch_time;
  fpses.push_back(fps);

  std::stringstream ss;
  ss << "Iteration: " << iter + 1;
  if (iter < skip_batch_num) ss << " (warm-up)";
  // use standard float formatting
  ss << std::fixed << std::setw(11) << std::setprecision(6);

  // first output: avg_cost
  if (output_slots.size() == 0)
    throw std::invalid_argument("Gather: output_slots vector is empty.");
  if (output_slots.size() >= 2UL) {  // acc_top1
    // second output: acc_top1
    if (output_slots[1].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: top1 accuracy output has nonempty LoD.");
    if (output_slots[1].dtype != paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "Gather: top1 accuracy output is of a wrong type.");
    float* acc1 = static_cast<float*>(output_slots[1].data.data());
    infer_accs1.push_back(*acc1);
    ss << ", accuracy: " << *acc1;
  }
  if (output_slots.size() >= 3UL) {  // acc_top5
    // third output: acc_top5
    if (output_slots[2].lod.size() > 0)
      throw std::invalid_argument(
          "Gather: top5 accuracy output has nonempty LoD.");
    if (output_slots[2].dtype != paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "Gather: top5 accuracy output is of a wrong type.");
    float* acc5 = static_cast<float*>(output_slots[2].data.data());
    infer_accs5.push_back(*acc5);
  }
  ss << ", latency: " << batch_time << " s, fps: " << fps;
  std::cout << ss.str() << std::endl;
}

// postprocess statistics from the whole test
void Stats::Postprocess(double total_time_sec, int total_samples) {
  SkipFirstNData(latencies, skip_batch_num);
  double lat_avg = FindAverage(latencies);
  double lat_pc99 = FindPercentile(latencies, 99);
  double lat_std = FindStandardDev(latencies);

  SkipFirstNData(fpses, skip_batch_num);
  double fps_avg = FindAverage(fpses);
  double fps_pc01 = FindPercentile(fpses, 1);
  double fps_std = FindStandardDev(fpses);

  float examples_per_sec = total_samples / total_time_sec;
  std::stringstream ss;
  // use standard float formatting
  ss << std::fixed << std::setw(11) << std::setprecision(6);
  ss << "\n";
  ss << "Avg fps: " << fps_avg << ", std fps: " << fps_std
     << ", fps for 99pc latency: " << fps_pc01 << std::endl;
  ss << "Avg latency: " << lat_avg << ", std latency: " << lat_std
     << ", 99pc latency: " << lat_pc99 << std::endl;
  ss << "Total examples: " << total_samples
     << ", total time: " << total_time_sec
     << ", total examples/sec: " << examples_per_sec << std::endl;

  if (infer_accs1.size() > 0) {
    SkipFirstNData(infer_accs1, skip_batch_num);
    float acc1_avg = FindAverage(infer_accs1);
    ss << "Avg top1 accuracy: " << acc1_avg << std::endl;
  }

  if (infer_accs5.size() > 0) {
    SkipFirstNData(infer_accs5, skip_batch_num);
    float acc5_avg = FindAverage(infer_accs5);
    ss << "Avg top5 accuracy: " << acc5_avg << std::endl;
  }
  std::cout << ss.str() << std::endl;
}

}  // namespace paddle
