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

#include "paddle/framework/executor.h"
#include "gtest/gtest.h"

using namespace paddle::platform;
using namespace paddle::framework;

TEST(Executor, Init) {
  CPUPlace cpu_place1, cpu_place2;
  std::vector<Place> places;
  places.push_back(cpu_place1);
  places.push_back(cpu_place2);
  Executor* executor = new Executor(places);

  ProgramDesc pdesc;
  Scope s;
  std::vector<Tensor>* outputs{nullptr};
  executor->Run(pdesc, &s, outputs);
  delete executor;
}