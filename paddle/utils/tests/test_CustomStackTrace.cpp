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

#include <chrono>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "paddle/utils/CustomStackTrace.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

DEFINE_int32(test_thread_num, 10, "testing thread number");

void testNormalImpl(
    const std::function<void(paddle::CustomStackTrace<std::string>&,
                             size_t,
                             size_t,
                             paddle::ThreadBarrier&,
                             paddle::ThreadBarrier&)>& callback) {
  paddle::CustomStackTrace<std::string> tracer;
  paddle::ThreadBarrier doneBarrier(FLAGS_test_thread_num + 1);
  paddle::ThreadBarrier startBarrier(FLAGS_test_thread_num + 1);
  constexpr size_t countDown = 10;
  constexpr size_t layerSize = 1000;
  std::vector<std::unique_ptr<std::thread>> threads;
  threads.reserve(FLAGS_test_thread_num);

  for (int32_t i = 0; i < FLAGS_test_thread_num; ++i) {
    threads.emplace_back(new std::thread([&tracer,
                                          &countDown,
                                          &layerSize,
                                          &startBarrier,
                                          &doneBarrier,
                                          &callback] {
      callback(tracer, countDown, layerSize, startBarrier, doneBarrier);
    }));
  }
  size_t cntDown = countDown;
  while (cntDown-- > 0) {
    startBarrier.wait();
    sleep(1);
    doneBarrier.wait();
    ASSERT_TRUE(tracer.empty());
  }

  for (auto& thread : threads) {
    thread->join();
  }
}

TEST(CustomStackTrace, normalTrain) {
  testNormalImpl([](paddle::CustomStackTrace<std::string>& tracer,
                    size_t countDown,
                    size_t layerSize,
                    paddle::ThreadBarrier& start,
                    paddle::ThreadBarrier& finish) {
    while (countDown-- > 0) {
      start.wait();
      for (size_t i = 0; i < layerSize; ++i) {
        tracer.push("layer_" + paddle::str::to_string(i));
      }
      tracer.pop("");
      for (size_t i = 0; i < layerSize; ++i) {
        tracer.pop("layer_" + paddle::str::to_string(layerSize - 1 - i));
      }
      finish.wait();
    }
  });
}

TEST(CustomStackTrace, normalTest) {
  testNormalImpl([](paddle::CustomStackTrace<std::string>& tracer,
                    size_t countDown,
                    size_t layerSize,
                    paddle::ThreadBarrier& start,
                    paddle::ThreadBarrier& finish) {
    while (countDown-- > 0) {
      start.wait();
      for (size_t i = 0; i < layerSize; ++i) {
        tracer.push("layer_" + paddle::str::to_string(i));
      }
      tracer.clear();  // in forward test, tracer will clear after forward.
      finish.wait();
    }
  });
}
