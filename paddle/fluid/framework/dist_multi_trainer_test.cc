//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/trainer.h"
#include "gtest/gtest.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {
TEST(DisMultiTrainerTest, test1) {
#ifdef _LINUX
  std::shared_ptr<DistMultiTrainer> tmp = std::make_shared<DistMultiTrainer>();
  TrainerDesc t;
  std::shared_ptr<MultiSlotDataset> dataset =
      std::make_shared<MultiSlotDataset>();
  dataset->SetFileList(std::vector<std::string>());
  dataset->SetThreadNum(1);
  dataset->SetTrainerNum(1);
  dataset->CreateReaders();
  tmp->Initialize(t, dataset.get());
#endif
}
}  // namespace framework
}  // namespace paddle
