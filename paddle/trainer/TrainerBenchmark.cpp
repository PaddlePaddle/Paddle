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

#undef PADDLE_DISABLE_TIMER

#include "Trainer.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

DECLARE_int32(test_period);

DEFINE_bool(feed_data, false, "Wether to read data from DataProvider.");

namespace paddle {

void Trainer::time() {
  startTrain();

  trainerInternal_.getParameterUpdater()->startPass();
  evaluator_->start();

  DataBatch dataBatch;
  int32_t batchSize = config_->getOptConfig().batch_size();
  int32_t num = dataProvider_->getNextBatch(batchSize, &dataBatch);
  CHECK_EQ(num, batchSize) << "The sample number is less than batch size "
                           << num << " != " << batchSize;

  CHECK(dataBatch.getSize()) << "No data from data provider";

  std::vector<paddle::Argument> outputs;
  // burning time
  LOG(INFO) << "Burning time...";
  for (int n = 0; n < 10; ++n) {
    trainerInternal_.trainOneBatch(n, dataBatch, &outputs);
  }
  LOG(INFO) << "Burning time end.";

  for (int n = 0; n < FLAGS_test_period; n++) {
    if (FLAGS_feed_data) {
      REGISTER_TIMER("GetData");
      num = dataProvider_->getNextBatch(batchSize, &dataBatch);
    }

    if (num != batchSize) {
      break;
    }

    {
      REGISTER_TIMER("FwdBwd");
      trainerInternal_.trainOneBatch(n, dataBatch, &outputs);
    }
  }
  globalStat.setThreadInfo(true);
  globalStat.printSegTimerStatus();
  globalStat.reset();

  finishTrain();
}

}  // namespace paddle
