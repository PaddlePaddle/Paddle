// Copyright 2018-2019, Mingkun Huang
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

#include <random>
#include <vector>

float* genActs(int size) {
  float* arr = new float[size];
  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(0, 1);
  for (int i = 0; i < size; ++i) arr[i] = dis(gen);
  return arr;
}

void genActs(std::vector<float>& arr) {
  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(0, 1);
  for (size_t i = 0; i < arr.size(); ++i) arr[i] = dis(gen);
}

std::vector<int> genLabels(int alphabet_size, int L) {
  std::vector<int> label(L);

  std::mt19937 gen(1);
  std::uniform_int_distribution<> dis(1, alphabet_size - 1);

  for (int i = 0; i < L; ++i) {
    label[i] = dis(gen);
  }
  // guarantee repeats for testing
  if (L >= 3) {
    label[L / 2] = label[L / 2 + 1];
    label[L / 2 - 1] = label[L / 2];
  }
  return label;
}
