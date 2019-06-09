/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

TEST(inference, label_semantic_roles) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  paddle::framework::LoDTensor word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1,
      ctx_p2, mark;
  paddle::framework::LoD lod{{0, 4, 10}};
  int64_t word_dict_len = 44068;
  int64_t predicate_dict_len = 3162;
  int64_t mark_dict_len = 2;

  SetupLoDTensor(&word, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&predicate, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(predicate_dict_len - 1));
  SetupLoDTensor(&ctx_n2, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&ctx_n1, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&ctx_0, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&ctx_p1, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&ctx_p2, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(word_dict_len - 1));
  SetupLoDTensor(&mark, lod, static_cast<int64_t>(0),
                 static_cast<int64_t>(mark_dict_len - 1));

  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&word);
  cpu_feeds.push_back(&predicate);
  cpu_feeds.push_back(&ctx_n2);
  cpu_feeds.push_back(&ctx_n1);
  cpu_feeds.push_back(&ctx_0);
  cpu_feeds.push_back(&ctx_p1);
  cpu_feeds.push_back(&ctx_p2);
  cpu_feeds.push_back(&mark);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  // Run inference on CPU
  TestInference<paddle::platform::CPUPlace>(dirname, cpu_feeds, cpu_fetchs1);
  LOG(INFO) << output1.lod();
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.lod();
  LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}
