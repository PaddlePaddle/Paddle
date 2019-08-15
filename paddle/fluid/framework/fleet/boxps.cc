// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/fleet/boxps.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

namespace paddle {
namespace boxps {
void FakeBoxPS::PrintAllEmb() const {
  for (auto i = emb_.begin(); i != emb_.end(); ++i) {
    printf("%lu: ", i->first);
    for (const auto e : i->second) {
      printf("%f ", e);
    }
    printf("\n");
  }
}

void FakeBoxPS::DebugPrintKey(const uint64_t *d, int len,
                              const std::string &info) const {
  printf("FakeBoxPS: %s\n", info.c_str());
  for (int i = 0; i < len; ++i) {
    printf("%lu ", d[i]);
  }
  printf("\n");
}

int FakeBoxPS::BeginPass(int date, const std::vector<uint64_t> &pass_data) {
  printf("FakeBoxPS: Pass begin...\n");
  printf("FakeBoxPS: date: %d\n", date);
  for (const auto fea : pass_data) {
    if (emb_.find(fea) == emb_.end()) {
      emb_[fea] = std::vector<float>(hidden_size_, 0.0);
    }
  }
  PrintAllEmb();
  return 0;
}

int FakeBoxPS::EndPass() {
  printf("FakeBoxPS: Pass end...\n");
  return 0;
}

int FakeBoxPS::InitializeCPU(const char *conf_file, int minibatch_size) {
  return 0;
}

int FakeBoxPS::PullSparseCPU(const uint64_t *keys, FeatureValue **vals,
                             int fea_num) {
  printf("FakeBoxPS:begin pull sparse...\n");
  this->DebugPrintKey(keys, fea_num, "key in pullsparse cpu:");

  *vals = new FeatureValue[fea_num];
  for (int i = 0; i < fea_num; ++i) {
    const auto iter = emb_.find(*(keys + i));
    if (iter == emb_.end()) {
      printf("Pull Sparse error - no key for %lu\n", *(keys + i));
      return 1;
    }
    const auto *value = iter->second.data();
    // memory in values has been allocated in pull_sparse_op

    printf("value: ");
    for (int j = 0; j < hidden_size_; ++j) {
      printf("%f ", value[j]);
    }
    printf("\n");
    memcpy(reinterpret_cast<float *>(&((*vals + i)->show)), value,
           hidden_size_ * sizeof(float));
    printf("FakeBoxPS: i:%d, show:%f, click:%f\n", i, (*vals + i)->show,
           (*vals + i)->clk);
  }
  printf("FakeBoxPS:end pull sparse...\n");
  return 0;
}

int FakeBoxPS::PushSparseCPU(const uint64_t *keys,
                             const FeaturePushValue *push_vals, int fea_num) {
  // should add lock for multi-thread
  printf("FakeBoxPS:begin push grad sparse...\n");
  PrintAllEmb();
  for (int i = 0; i < fea_num; ++i) {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto iter = emb_.find(*(keys + i));
    if (iter == emb_.end()) {
      printf("Push Sparse grad error - no key for %lu\n", *(keys + i));
      return 1;
    }
    auto &para = iter->second;
    auto start_ptr = &((push_vals + i)->show);
    for (int j = 0; j < hidden_size_; ++j) {
      para[j] -= learning_rate_ * (*(start_ptr + j));
    }
  }
  printf("FakeBoxPS: end push grad sparse...\n");
  PrintAllEmb();
  return 0;
}

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
int FakeBoxPS::InitializeGPU(const char *conf_file, int minibatch_size,
                             const std::vector<cudaStream_t *> &stream_list) {
  return 0;
}

int FakeBoxPS::PullSparseGPU(const uint64_t *keys, FeatureValue **vals,
                             int fea_num, int stream_idx) {
  printf("FakeBoxPS PullSparseGPU begin\n");
  uint64_t *cpu_keys = new uint64_t[fea_num];
  cudaMemcpy(cpu_keys, keys, sizeof(uint64_t) * fea_num,
             cudaMemcpyDeviceToHost);

  // Debug info, should be deleted
  this->DebugPrintKey(cpu_keys, fea_num, "copy keys from gpu to cpu success");

  FeatureValue *cpu_values;
  this->PullSparseCPU(cpu_keys, &cpu_values, fea_num);
  cudaMemcpy(vals, cpu_values, sizeof(FeatureValue) * fea_num,
             cudaMemcpyHostToDevice);
  delete[] cpu_keys;
  delete[] cpu_values;
  return 0;
}

int FakeBoxPS::PushSparseGPU(const uint64_t *keys,
                             const FeaturePushValue *push_vals, int fea_num,
                             int stream_idx) {
  printf("FakeBoxPS PushSparseGPU begin\n");
  uint64_t *cpu_keys = new uint64_t[fea_num];
  cudaMemcpy(cpu_keys, keys, sizeof(uint64_t) * fea_num,
             cudaMemcpyDeviceToHost);

  // Debug, should delete
  this->DebugPrintKey(cpu_keys, fea_num, "copy keys from gpu to cpu success");

  FeaturePushValue *cpu_grad_values = new FeaturePushValue[fea_num];
  cudaMemcpy(cpu_grad_values, push_vals, sizeof(FeaturePushValue) * fea_num,
             cudaMemcpyDeviceToHost);

  this->PushSparseCPU(cpu_keys, cpu_grad_values, fea_num);
  delete[] cpu_keys;
  delete[] cpu_grad_values;
  return 0;
}
#endif
int FakeBoxPS::LoadModel(const std::string &path, const int mode) { return 0; }

int FakeBoxPS::SaveModel(const std::string &path, const int mode) { return 0; }
}  // end namespace boxps
}  // end namespace paddle
