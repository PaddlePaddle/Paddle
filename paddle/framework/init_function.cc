/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/framework/enforce.h>
#include <paddle/framework/init_function.h>
#include <algorithm>
#include <map>
#include <vector>
namespace paddle {
namespace framework {

using PriorityFuncPair = std::pair<int, InitFunctionType>;
static bool g_initialized = false;
static std::vector<PriorityFuncPair>* g_priority_func_pairs = nullptr;
void RegisterInitFunction(InitFunctionType func, int priority) {
  if (g_priority_func_pairs == nullptr) {
    g_priority_func_pairs = new std::vector<PriorityFuncPair>();
  }
  PADDLE_ENFORCE(!g_initialized,
                 "RegisterInitFunction() should only called before initMain()");
  g_priority_func_pairs->push_back({priority, func});
}

void RunInitFunctions() {
  if (g_initialized) return;
  PADDLE_ENFORCE(
      g_priority_func_pairs != nullptr,
      "RegisterInitFunction should be invoked before RunInitFunctions");
  std::sort(g_priority_func_pairs->begin(), g_priority_func_pairs->end(),
            [](const PriorityFuncPair& a, const PriorityFuncPair& b) {
              return a.first > b.first;
            });
  for (auto& func : *g_priority_func_pairs) {
    func.second();
  }
  delete g_priority_func_pairs;
  g_initialized = true;
}

}  // namespace framework
}  // namespace paddle