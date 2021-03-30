/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_mkldnn_quantizer_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

std::shared_ptr<std::vector<PaddleTensor>> GetDummyWarmupData() {
  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(1);
  PaddleTensor images;
  images.name = "image";
  images.shape = {2, 3, 300, 300};
  images.dtype = PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * 2 * 3 * 300 * 300);

  (*warmup_data)[0] = std::move(images);
  return warmup_data;
}

TEST(Mkldnn_quantizer_config, configuration) {
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data = GetDummyWarmupData();
  int warmup_data_size = 1;
  int warmup_batch_size = 2;
  std::unordered_set<std::string> enabled_op_types(
      {"conv2d", "fc", "reshape2"});
  std::unordered_set<int> excluded_op_ids({2, 3});
  ScaleAlgo default_scale_algo = ScaleAlgo::NONE;
  ScaleAlgo conv2d_scale_algo = ScaleAlgo::MAX;

  AnalysisConfig cfg;
  cfg.EnableMkldnnQuantizer();
  cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(warmup_batch_size);
  cfg.mkldnn_quantizer_config()->SetEnabledOpTypes(enabled_op_types);
  cfg.mkldnn_quantizer_config()->SetExcludedOpIds(excluded_op_ids);
  cfg.mkldnn_quantizer_config()->SetDefaultScaleAlgo(default_scale_algo);
  cfg.mkldnn_quantizer_config()->SetScaleAlgo("conv2d", "Input",
                                              conv2d_scale_algo);

  PADDLE_ENFORCE_EQ(cfg.mkldnn_quantizer_config()->warmup_data()->size(),
                    warmup_data_size,
                    platform::errors::InvalidArgument(
                        "Size of the warmup data got from config differs with "
                        "the one set previously."));

  PADDLE_ENFORCE_EQ(
      cfg.mkldnn_quantizer_config()->warmup_data()->at(0).name, "image",
      platform::errors::InvalidArgument(
          "Warmup data got from config differs with the one set previously."));

  PADDLE_ENFORCE_EQ(cfg.mkldnn_quantizer_config()->warmup_batch_size(),
                    warmup_batch_size,
                    platform::errors::InvalidArgument(
                        "Warmup batch size got from config differs with the "
                        "one set previously."));

  PADDLE_ENFORCE_EQ(cfg.mkldnn_quantizer_config()->enabled_op_types(),
                    enabled_op_types,
                    platform::errors::InvalidArgument(
                        "Enabled op types list got from config differs with "
                        "the one set previously."));

  PADDLE_ENFORCE_EQ(cfg.mkldnn_quantizer_config()->excluded_op_ids(),
                    excluded_op_ids,
                    platform::errors::InvalidArgument(
                        "Excluded op ids list got from config differs with "
                        "the one set previously."));

  PADDLE_ENFORCE_EQ(cfg.mkldnn_quantizer_config()->default_scale_algo(),
                    default_scale_algo,
                    platform::errors::InvalidArgument(
                        "Default scale algorithm got from config differs with "
                        "the one set previously."));

  PADDLE_ENFORCE_EQ(
      cfg.mkldnn_quantizer_config()->scale_algo("conv2d", "Input"),
      conv2d_scale_algo, platform::errors::InvalidArgument(
                             "Scale algorithm got from config differs with the "
                             "one set previously."));

  PADDLE_ENFORCE_EQ(
      cfg.mkldnn_quantizer_config()->scale_algo("unknown", "unknown"),
      default_scale_algo,
      platform::errors::InvalidArgument(
          "Scale algorithm got from config for an uknown op "
          "differs with the one set previously."));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
