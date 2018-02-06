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
    EXPECT_NE(dynamic_cast<Buffered<int> *>(ch), nullptr);
    EXPECT_EQ(dynamic_cast<UnBuffered<int> *>(ch), nullptr);
    CloseChannel(ch);
    delete ch;
  }
  {
    // MakeChannel should return an un-buffered channel is buffer_size = 0.
    auto ch = MakeChannel<int>(0);
    EXPECT_EQ(dynamic_cast<Buffered<int> *>(ch), nullptr);
    EXPECT_NE(dynamic_cast<UnBuffered<int> *>(ch), nullptr);
    CloseChannel(ch);
    delete ch;
  }
}

TEST(Channel, SufficientBufferSizeDoesntBlock) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ch->Send(&i), true);  // should not block
  }

  size_t out;
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ch->Receive(&out), true);  // should not block
    EXPECT_EQ(out, i);
  }
  CloseChannel(ch);
  delete ch;
}

TEST(Channel, ConcurrentSendNonConcurrentReceiveWithSufficientBufferSize) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  size_t sum = 0;
  std::thread t([&]() {
    // Try to write more than buffer size.
    for (size_t i = 0; i < 2 * buffer_size; ++i) {
      if (i < buffer_size)
        EXPECT_EQ(ch->Send(&i), true);  // should block after 10 iterations
      else
        EXPECT_EQ(ch->Send(&i), false);
      sum += i;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait 0.5 sec
  EXPECT_EQ(sum, 45U);

  CloseChannel(ch);
  t.join();
  delete ch;
}

TEST(Channel, SimpleUnbufferedChannelTest) {
  auto ch = MakeChannel<int>(0);
  unsigned sum_send = 0;
  std::thread t([&]() {
    for (int i = 0; i < 5; i++) {
      EXPECT_EQ(ch->Send(&i), true);
      sum_send += i;
    }
  });
  for (int i = 0; i < 5; i++) {
    int recv;
    EXPECT_EQ(ch->Receive(&recv), true);
    EXPECT_EQ(recv, i);
  }

  CloseChannel(ch);
  t.join();
  EXPECT_EQ(sum_send, 10U);
  delete ch;
}

// This tests that closing a buffered channel also unblocks
//  any receivers waiting on the channel
TEST(Channel, BufferedChannelCloseUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(1);
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          // All reads should return false
          EXPECT_EQ(ch->Receive(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait

  // Verify that all threads are blocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }

  // Explicitly close the channel
  // This should unblock all receivers
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < num_threads; i++) t[i].join();
  delete ch;
}

// This tests that closing a buffered channel also unblocks
//  any senders waiting for channel to have write space
TEST(Channel, BufferedChannelCloseUnblocksSendersTest) {
  auto ch = MakeChannel<int>(1);
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];
  bool send_success[num_threads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          *success = ch->Send(&data);
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait

  // Verify that atleast 4 threads are blocked
  int ct = 0;
  for (size_t i = 0; i < num_threads; i++) {
    if (thread_ended[i] == false) ct++;
  }
  // Atleast 4 threads must be blocked
  EXPECT_GE(ct, 4);

  // Explicitly close the thread
  // This should unblock all senders
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  // Verify that only 1 send was successful
  ct = 0;
  for (size_t i = 0; i < num_threads; i++) {
    if (send_success[i]) ct++;
  }
  // Only 1 send must be successful
  EXPECT_EQ(ct, 1);

  for (size_t i = 0; i < num_threads; i++) t[i].join();
  delete ch;
}

// This tests that closing an unbuffered channel also unblocks
//  unblocks any receivers waiting for senders
TEST(Channel, UnbufferedChannelCloseUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(0);
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];

  // Launches threads that try to read and are blocked becausew of no writers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          EXPECT_EQ(ch->Receive(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // wait 0.5 sec

  // Verify that all the threads are blocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }

  // Explicitly close the thread
  // This should unblock all receivers
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // wait 0.5 sec

  // Verify that all threads got unblocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < num_threads; i++) t[i].join();
  delete ch;
}

// This tests that closing an unbuffered channel also unblocks
//  unblocks any senders waiting for senders
TEST(Channel, UnbufferedChannelCloseUnblocksSendersTest) {
  auto ch = MakeChannel<int>(0);
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];

  // Launches threads that try to read and are blocked becausew of no writers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data = 10;
          EXPECT_EQ(ch->Send(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // wait 0.5 sec

  // Verify that all the threads are blocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }

  // Explicitly close the thread
  // This should unblock all receivers
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // wait 0.5 sec

  // Verify that all threads got unblocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < num_threads; i++) t[i].join();
  delete ch;
}

TEST(Channel, UnbufferedLessReceiveMoreSendTest) {
  auto ch = MakeChannel<int>(0);
  unsigned sum_send = 0;
  // Send should block after three iterations
  // since we only have three receivers.
  std::thread t([&]() {
    // Try to send more number of times
    // than receivers
    for (int i = 0; i < 4; i++) {
      ch->Send(&i);
      sum_send += i;
    }
  });
  for (int i = 0; i < 3; i++) {
    int recv;
    ch->Receive(&recv);
    EXPECT_EQ(recv, i);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait 0.5 sec
  EXPECT_EQ(sum_send, 3U);

  CloseChannel(ch);
  t.join();
  delete ch;
}

TEST(Channel, UnbufferedMoreReceiveLessSendTest) {
  auto ch = MakeChannel<int>(0);
  unsigned sum_send = 0;
  unsigned sum_receive = 0;
  // The receiver should block after 5
  // iterations, since there are only 5 senders.
  std::thread t([&]() {
    for (int i = 0; i < 8; i++) {
      int recv;
      ch->Receive(&recv);  // should block after the fifth iteration.
      EXPECT_EQ(recv, i);
      sum_receive += i;
    }
  });
  for (int i = 0; i < 5; i++) {
    ch->Send(&i);
    sum_send += i;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // wait 0.5 sec
  EXPECT_EQ(sum_send, 10U);
  EXPECT_EQ(sum_receive, 10U);
  // send three more elements
  for (int i = 5; i < 8; i++) {
    ch->Send(&i);
    sum_send += i;
  }

  CloseChannel(ch);
  t.join();
  EXPECT_EQ(sum_send, 28U);
  EXPECT_EQ(sum_receive, 28U);
  delete ch;
}
