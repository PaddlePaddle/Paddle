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

#include "paddle/framework/channel.h"

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

using paddle::framework::Channel;
using paddle::framework::MakeChannel;
using paddle::framework::CloseChannel;

TEST(Channel, MakeAndClose) {
  using paddle::framework::details::Buffered;
  using paddle::framework::details::UnBuffered;
  {
    // MakeChannel should return a buffered channel is buffer_size > 0.
    auto ch = MakeChannel<int>(10);
    EXPECT_NE(dynamic_cast<Buffered<int>*>(ch), nullptr);
    EXPECT_EQ(dynamic_cast<UnBuffered<int>*>(ch), nullptr);
    CloseChannel(ch);
  }
  {
    // MakeChannel should return an un-buffered channel is buffer_size = 0.
    auto ch = MakeChannel<int>(0);
    EXPECT_EQ(dynamic_cast<Buffered<int>*>(ch), nullptr);
    EXPECT_NE(dynamic_cast<UnBuffered<int>*>(ch), nullptr);
    CloseChannel(ch);
  }
}

TEST(Channel, SufficientBufferSizeDoesntBlock) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  for (size_t i = 0; i < buffer_size; ++i) {
    ch->Send(&i);  // should not block
  }

  size_t out;
  for (size_t i = 0; i < buffer_size; ++i) {
    ch->Receive(&out);  // should not block
    EXPECT_EQ(out, i);
  }
  CloseChannel(ch);
}

TEST(Channel, ConcurrentSendNonConcurrentReceiveWithSufficientBufferSize) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);

  size_t sum = 0;
  std::thread t([&]() {
    // Try to write more than buffer size.
    for (size_t i = 0; i < 2 * buffer_size; ++i) {
      ch->Send(&i);  // should not block
      sum += i;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait 0.5 sec
  EXPECT_EQ(sum, 45U);
  CloseChannel(ch);
}
