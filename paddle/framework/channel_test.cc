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
#include <gtest/gtest.h>

#include "paddle/framework/channel.h"
#include "threadpool.h"

using paddle::framework::Channel;
using paddle::framework::MakeChannel;
using paddle::framework::CloseChannel;

TEST(Channel, MakeAndClose) {
  Channel<int>* ch = MakeChannel<int>(10);
  CloseChannel(ch);
}

TEST(Channel, Buffered) {
  Channel<int>* ch = MakeChannel<int>(10);

  for (int i = 0; i < 10; ++i) {
    ch->Send(&i);
  }

  int temp = -1;
  for (int j = 0; j < 10; ++j) {
    ch->Receive(&temp);
    EXPECT_EQ(temp, j);
  }
  CloseChannel(ch);
}

TEST(Channel, MultiThread) {
  namespace framework = paddle::framework;

  const int capacity = 10;
  Channel<int>* ch = MakeChannel<int>(capacity);

  framework::ThreadPool* pool;
  pool = framework::ThreadPool::GetInstance();

  // Consumer
  for (int i = 0; i < capacity; ++i) {
    framework::Async([&ch]() {
      int temp = -1;
      ch->Receive(&temp);
      EXPECT_GE(temp, -1);
    });
  }

  // Producer
  for (int i = 0; i < capacity; ++i) {
    framework::Async([&ch, &i]() { ch->Send(&i); });
  }

  pool->Wait();
  CloseChannel(ch);
}
