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

#include "paddle/fluid/framework/channel.h"

#include <chrono>  // NOLINT
#include <thread>  // NOLINT
#include "gtest/gtest.h"

using paddle::framework::Channel;
using paddle::framework::ChannelHolder;
using paddle::framework::MakeChannel;
using paddle::framework::CloseChannel;

TEST(Channel, ChannelCapacityTest) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  EXPECT_EQ(ch->Cap(), buffer_size);
  CloseChannel(ch);
  delete ch;

  ch = MakeChannel<size_t>(0);
  EXPECT_EQ(ch->Cap(), 0U);
  CloseChannel(ch);
  delete ch;
}

void RecevingOrderEqualToSendingOrder(Channel<int> *ch, int num_items) {
  unsigned sum_send = 0;
  std::thread t([&]() {
    for (int i = 0; i < num_items; i++) {
      ch->Send(&i);
      sum_send += i;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 0; i < num_items; i++) {
    int recv = -1;
    EXPECT_EQ(ch->Receive(&recv), true);
    EXPECT_EQ(recv, i);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CloseChannel(ch);
  t.join();
  unsigned expected_sum = (num_items * (num_items - 1)) / 2;
  EXPECT_EQ(sum_send, expected_sum);
  delete ch;
}

TEST(Channel, SufficientBufferSizeDoesntBlock) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  for (size_t i = 0; i < buffer_size; ++i) {
    ch->Send(&i);
  }

  size_t out;
  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ch->Receive(&out), true);  // should not block
    EXPECT_EQ(out, i);
  }
  CloseChannel(ch);
  delete ch;
}

// This tests that a  channel must return false
// on send and receive performed after closing the channel.
// Receive will only return false after close when queue is empty.
// By creating separate threads for sending and receiving, we make this
// function able to test both buffered and unbuffered channels.
void SendReceiveWithACloseChannelShouldPanic(Channel<size_t> *ch) {
  const size_t data = 5;
  std::thread send_thread{[&]() {
    size_t i = data;
    ch->Send(&i);  // should not block
  }};

  std::thread recv_thread{[&]() {
    size_t i;
    EXPECT_EQ(ch->Receive(&i), true);  // should not block
    EXPECT_EQ(i, data);
  }};

  send_thread.join();
  recv_thread.join();

  // After closing send should panic. Receive should
  // also  false as there is no data in queue.
  CloseChannel(ch);
  send_thread = std::thread{[&]() {
    size_t i = data;
    bool is_exception = false;
    try {
      ch->Send(&i);
    } catch (paddle::platform::EnforceNotMet e) {
      is_exception = true;
    }
    EXPECT_EQ(is_exception, true);
  }};
  recv_thread = std::thread{[&]() {
    size_t i;
    // should return false because channel is closed and queue is empty
    EXPECT_EQ(ch->Receive(&i), false);
  }};

  send_thread.join();
  recv_thread.join();
}

TEST(Channel, SendReceiveClosedBufferedChannelPanics) {
  size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  SendReceiveWithACloseChannelShouldPanic(ch);
  delete ch;
}

TEST(Channel, SendReceiveClosedUnBufferedChannelPanics) {
  auto ch = MakeChannel<size_t>(0);
  SendReceiveWithACloseChannelShouldPanic(ch);
  delete ch;
}

TEST(Channel, ReceiveFromBufferedChannelReturnResidualValuesTest) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);

  for (size_t i = 0; i < buffer_size; ++i) {
    ch->Send(&i);  // sending should not block
  }

  size_t out;
  for (size_t i = 0; i < buffer_size / 2; ++i) {
    EXPECT_EQ(ch->Receive(&out), true);  // receiving should not block
    EXPECT_EQ(out, i);
  }

  CloseChannel(ch);

  for (size_t i = buffer_size / 2; i < buffer_size; ++i) {
    EXPECT_EQ(ch->Receive(&out),
              true);  // receving should return residual values.
    EXPECT_EQ(out, i);
  }

  for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(ch->Receive(&out),
              false);  // receiving on closed channel should return false
  }
  delete ch;
}

TEST(Channel, ConcurrentSendNonConcurrentReceiveWithSufficientBufferSize) {
  const size_t buffer_size = 10;
  auto ch = MakeChannel<size_t>(buffer_size);
  std::thread t([&]() {
    // Try to write more than buffer size.
    for (size_t i = 0; i < 2 * buffer_size; ++i) {
      if (i < buffer_size) {
        ch->Send(&i);  // should block after 10 iterations
      } else {
        bool is_exception = false;
        try {
          ch->Send(&i);
        } catch (paddle::platform::EnforceNotMet e) {
          is_exception = true;
        }
        EXPECT_EQ(is_exception, true);
      }
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec
  CloseChannel(ch);
  t.join();
  delete ch;
}

TEST(Channel, RecevingOrderEqualToSendingOrderWithUnBufferedChannel) {
  auto ch = MakeChannel<int>(0);
  RecevingOrderEqualToSendingOrder(ch, 20);
}

TEST(Channel, RecevingOrderEqualToSendingOrderWithBufferedChannel1) {
  // Test that Receive Order is same as Send Order when number of items
  // sent is less than size of buffer
  auto ch = MakeChannel<int>(10);
  RecevingOrderEqualToSendingOrder(ch, 5);
}

TEST(Channel, RecevingOrderEqualToSendingOrderWithBufferedChannel2) {
  // Test that Receive Order is same as Send Order when number of items
  // sent is equal to size of buffer
  auto ch = MakeChannel<int>(10);
  RecevingOrderEqualToSendingOrder(ch, 10);
}

TEST(Channel, RecevingOrderEqualToSendingOrderWithBufferedChannel3) {
  // Test that Receive Order is same as Send Order when number of items
  // sent is greater than the size of buffer
  auto ch = MakeChannel<int>(10);
  RecevingOrderEqualToSendingOrder(ch, 20);
}

void ChannelCloseUnblocksReceiversTest(Channel<int> *ch) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          EXPECT_EQ(ch->Receive(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  // Verify that all the threads are blocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }

  // Explicitly close the channel
  // This should unblock all receivers
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

void ChannelCloseUnblocksSendersTest(Channel<int> *ch, bool isBuffered) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];
  bool send_success[kNumThreads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          bool is_exception = false;
          try {
            ch->Send(&data);
          } catch (paddle::platform::EnforceNotMet e) {
            is_exception = true;
          }
          *success = !is_exception;
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  if (isBuffered) {
    // If ch is Buffered, atleast 4 threads must be blocked.
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (!thread_ended[i]) ct++;
    }
    EXPECT_GE(ct, 4);
  } else {
    // If ch is UnBuffered, all the threads should be blocked.
    for (size_t i = 0; i < kNumThreads; i++) {
      EXPECT_EQ(thread_ended[i], false);
    }
  }
  // Explicitly close the thread
  // This should unblock all senders
  CloseChannel(ch);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  if (isBuffered) {
    // Verify that only 1 send was successful
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (send_success[i]) ct++;
    }
    // Only 1 send must be successful
    EXPECT_EQ(ct, 1);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

// This tests that closing a buffered channel also unblocks
//  any receivers waiting on the channel
TEST(Channel, BufferedChannelCloseUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(1);
  ChannelCloseUnblocksReceiversTest(ch);
  delete ch;
}

// This tests that closing a buffered channel also unblocks
//  any senders waiting for channel to have write space
TEST(Channel, BufferedChannelCloseUnblocksSendersTest) {
  auto ch = MakeChannel<int>(1);
  ChannelCloseUnblocksSendersTest(ch, true);
  delete ch;
}

// This tests that closing an unbuffered channel also unblocks
//  unblocks any receivers waiting for senders
TEST(Channel, UnbufferedChannelCloseUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(0);
  ChannelCloseUnblocksReceiversTest(ch);
  delete ch;
}

// This tests that closing an unbuffered channel also unblocks
//  unblocks any senders waiting for senders
TEST(Channel, UnbufferedChannelCloseUnblocksSendersTest) {
  auto ch = MakeChannel<int>(0);
  ChannelCloseUnblocksSendersTest(ch, false);
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
      try {
        ch->Send(&i);
        sum_send += i;
      } catch (paddle::platform::EnforceNotMet e) {
      }
    }
  });
  for (int i = 0; i < 3; i++) {
    int recv;
    ch->Receive(&recv);
    EXPECT_EQ(recv, i);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec
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
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec
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

// This tests that destroying a channel unblocks
//  any senders waiting for channel to have write space
void ChannelDestroyUnblockSenders(Channel<int> *ch, bool isBuffered) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];
  bool send_success[kNumThreads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          bool is_exception = false;
          try {
            ch->Send(&data);
          } catch (paddle::platform::EnforceNotMet e) {
            is_exception = true;
          }
          *success = !is_exception;
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  if (isBuffered) {
    // If channel is buffered, verify that atleast 4 threads are blocked
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (thread_ended[i] == false) ct++;
    }
    // Atleast 4 threads must be blocked
    EXPECT_GE(ct, 4);
  } else {
    // Verify that all the threads are blocked
    for (size_t i = 0; i < kNumThreads; i++) {
      EXPECT_EQ(thread_ended[i], false);
    }
  }
  // Explicitly destroy the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  // Count number of successful sends
  int ct = 0;
  for (size_t i = 0; i < kNumThreads; i++) {
    if (send_success[i]) ct++;
  }

  if (isBuffered) {
    // Only 1 send must be successful
    EXPECT_EQ(ct, 1);
  } else {
    // In unbuffered channel, no send should be successful
    EXPECT_EQ(ct, 0);
  }

  // Join all threads
  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

// This tests that destroying a channel also unblocks
//  any receivers waiting on the channel
void ChannelDestroyUnblockReceivers(Channel<int> *ch) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < kNumThreads; i++) {
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
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }
  // delete the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait
  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

TEST(Channel, BufferedChannelDestroyUnblocksReceiversTest) {
  size_t buffer_size = 1;
  auto ch = MakeChannel<int>(buffer_size);
  ChannelDestroyUnblockReceivers(ch);
}

TEST(Channel, BufferedChannelDestroyUnblocksSendersTest) {
  size_t buffer_size = 1;
  auto ch = MakeChannel<int>(buffer_size);
  ChannelDestroyUnblockSenders(ch, true);
}

// This tests that destroying an unbuffered channel also unblocks
//  unblocks any receivers waiting for senders
TEST(Channel, UnbufferedChannelDestroyUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(0);
  ChannelDestroyUnblockReceivers(ch);
}

TEST(Channel, UnbufferedChannelDestroyUnblocksSendersTest) {
  auto ch = MakeChannel<int>(0);
  ChannelDestroyUnblockSenders(ch, false);
}

TEST(ChannelHolder, ChannelHolderCapacityTest) {
  const size_t buffer_size = 10;
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(buffer_size);
  EXPECT_EQ(ch->Cap(), buffer_size);
  delete ch;

  ch = new ChannelHolder();
  ch->Reset<int>(0);
  EXPECT_EQ(ch->Cap(), 0U);
  delete ch;
}

void ChannelHolderSendReceive(ChannelHolder *ch) {
  unsigned sum_send = 0;
  std::thread t([&]() {
    for (int i = 0; i < 5; i++) {
      ch->Send(&i);
      sum_send += i;
    }
  });
  for (int i = 0; i < 5; i++) {
    int recv;
    EXPECT_EQ(ch->Receive(&recv), true);
    EXPECT_EQ(recv, i);
  }

  ch->close();
  t.join();
  EXPECT_EQ(sum_send, 10U);
}

TEST(ChannelHolder, ChannelHolderBufferedSendReceiveTest) {
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(10);
  ChannelHolderSendReceive(ch);
  delete ch;
}

TEST(ChannelHolder, ChannelHolderUnBufferedSendReceiveTest) {
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(0);
  ChannelHolderSendReceive(ch);
  delete ch;
}

TEST(ChannelHolder, ChannelUninitializedTest) {
  ChannelHolder *ch = new ChannelHolder();
  EXPECT_EQ(ch->IsInitialized(), false);
  int i = 10;
  bool send_exception = false;
  try {
    ch->Send(&i);
  } catch (paddle::platform::EnforceNotMet e) {
    send_exception = true;
  }
  EXPECT_EQ(send_exception, true);

  bool recv_exception = false;
  try {
    ch->Receive(&i);
  } catch (paddle::platform::EnforceNotMet e) {
    recv_exception = true;
  }
  EXPECT_EQ(recv_exception, true);

  bool is_exception = false;
  try {
    ch->Type();
  } catch (paddle::platform::EnforceNotMet e) {
    is_exception = true;
  }
  EXPECT_EQ(is_exception, true);
  delete ch;
}

TEST(ChannelHolder, ChannelInitializedTest) {
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(2);
  EXPECT_EQ(ch->IsInitialized(), true);
  // Channel should remain intialized even after close
  ch->close();
  EXPECT_EQ(ch->IsInitialized(), true);
  delete ch;
}

TEST(ChannelHolder, TypeMismatchSendTest) {
  // Test with unbuffered channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(0);
  bool is_exception = false;
  bool boolean_data = true;
  try {
    ch->Send(&boolean_data);
  } catch (paddle::platform::EnforceNotMet e) {
    is_exception = true;
  }
  EXPECT_EQ(is_exception, true);
  delete ch;

  // Test with Buffered Channel
  ch = new ChannelHolder();
  ch->Reset<float>(10);
  is_exception = false;
  int int_data = 23;
  try {
    ch->Send(&int_data);
  } catch (paddle::platform::EnforceNotMet e) {
    is_exception = true;
  }
  EXPECT_EQ(is_exception, true);
  delete ch;
}

TEST(ChannelHolder, TypeMismatchReceiveTest) {
  // Test with unbuffered channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(0);
  bool is_exception = false;
  bool float_data;
  try {
    ch->Receive(&float_data);
  } catch (paddle::platform::EnforceNotMet e) {
    is_exception = true;
  }
  EXPECT_EQ(is_exception, true);
  delete ch;

  // Test with Buffered Channel
  ch = new ChannelHolder();
  ch->Reset<float>(10);
  is_exception = false;
  int int_data = 23;
  try {
    ch->Receive(&int_data);
  } catch (paddle::platform::EnforceNotMet e) {
    is_exception = true;
  }
  EXPECT_EQ(is_exception, true);
  delete ch;
}

void ChannelHolderCloseUnblocksReceiversTest(ChannelHolder *ch) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          EXPECT_EQ(ch->Receive(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  // Verify that all the threads are blocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }

  // Explicitly close the channel
  // This should unblock all receivers
  ch->close();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

void ChannelHolderCloseUnblocksSendersTest(ChannelHolder *ch, bool isBuffered) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];
  bool send_success[kNumThreads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          bool is_exception = false;
          try {
            ch->Send(&data);
          } catch (paddle::platform::EnforceNotMet e) {
            is_exception = true;
          }
          *success = !is_exception;
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  if (isBuffered) {
    // If ch is Buffered, atleast 4 threads must be blocked.
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (!thread_ended[i]) ct++;
    }
    EXPECT_GE(ct, 4);
  } else {
    // If ch is UnBuffered, all the threads should be blocked.
    for (size_t i = 0; i < kNumThreads; i++) {
      EXPECT_EQ(thread_ended[i], false);
    }
  }
  // Explicitly close the thread
  // This should unblock all senders
  ch->close();

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  if (isBuffered) {
    // Verify that only 1 send was successful
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (send_success[i]) ct++;
    }
    // Only 1 send must be successful
    EXPECT_EQ(ct, 1);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

// This tests that closing a channelholder unblocks
//  any receivers waiting on the channel
TEST(ChannelHolder, ChannelHolderCloseUnblocksReceiversTest) {
  // Check for buffered channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(1);
  ChannelHolderCloseUnblocksReceiversTest(ch);
  delete ch;

  // Check for unbuffered channel
  ch = new ChannelHolder();
  ch->Reset<int>(0);
  ChannelHolderCloseUnblocksReceiversTest(ch);
  delete ch;
}

// This tests that closing a channelholder unblocks
//  any senders waiting for channel to have write space
TEST(Channel, ChannelHolderCloseUnblocksSendersTest) {
  // Check for buffered channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(1);
  ChannelHolderCloseUnblocksSendersTest(ch, true);
  delete ch;

  // Check for unbuffered channel
  ch = new ChannelHolder();
  ch->Reset<int>(0);
  ChannelHolderCloseUnblocksSendersTest(ch, false);
  delete ch;
}

// This tests that destroying a channelholder unblocks
//  any senders waiting for channel
void ChannelHolderDestroyUnblockSenders(ChannelHolder *ch, bool isBuffered) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];
  bool send_success[kNumThreads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < kNumThreads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          bool is_exception = false;
          try {
            ch->Send(&data);
          } catch (paddle::platform::EnforceNotMet e) {
            is_exception = true;
          }
          *success = !is_exception;
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec
  if (isBuffered) {
    // If channel is buffered, verify that atleast 4 threads are blocked
    int ct = 0;
    for (size_t i = 0; i < kNumThreads; i++) {
      if (thread_ended[i] == false) ct++;
    }
    // Atleast 4 threads must be blocked
    EXPECT_GE(ct, 4);
  } else {
    // Verify that all the threads are blocked
    for (size_t i = 0; i < kNumThreads; i++) {
      EXPECT_EQ(thread_ended[i], false);
    }
  }
  // Explicitly destroy the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  // Count number of successfuld sends
  int ct = 0;
  for (size_t i = 0; i < kNumThreads; i++) {
    if (send_success[i]) ct++;
  }

  if (isBuffered) {
    // Only 1 send must be successful
    EXPECT_EQ(ct, 1);
  } else {
    // In unbuffered channel, no send should be successful
    EXPECT_EQ(ct, 0);
  }

  // Join all threads
  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

// This tests that destroying a channelholder also unblocks
//  any receivers waiting on the channel
void ChannelHolderDestroyUnblockReceivers(ChannelHolder *ch) {
  const size_t kNumThreads = 5;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < kNumThreads; i++) {
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
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Verify that all threads are blocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }
  // delete the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait
  // Verify that all threads got unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

TEST(ChannelHolder, ChannelHolderDestroyUnblocksReceiversTest) {
  // Check for Buffered Channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(1);
  ChannelHolderDestroyUnblockReceivers(ch);
  // ch is already deleted already deleted in
  // ChannelHolderDestroyUnblockReceivers

  // Check for Unbuffered channel
  ch = new ChannelHolder();
  ch->Reset<int>(0);
  ChannelHolderDestroyUnblockReceivers(ch);
}

TEST(ChannelHolder, ChannelHolderDestroyUnblocksSendersTest) {
  // Check for Buffered Channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(1);
  ChannelHolderDestroyUnblockSenders(ch, true);
  // ch is already deleted already deleted in
  // ChannelHolderDestroyUnblockReceivers

  // Check for Unbuffered channel
  ch = new ChannelHolder();
  ch->Reset<int>(0);
  ChannelHolderDestroyUnblockSenders(ch, false);
}

// This tests that closing a channelholder many times.
void ChannelHolderManyTimesClose(ChannelHolder *ch) {
  const int kNumThreads = 15;
  std::thread t[kNumThreads];
  bool thread_ended[kNumThreads];

  // Launches threads that try to send data to channel.
  for (size_t i = 0; i < kNumThreads / 3; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *ended) {
          int data = 10;
          ch->Send(&data);
          *ended = true;
        },
        &thread_ended[i]);
  }

  // Launches threads that try to receive data to channel.
  for (size_t i = kNumThreads / 3; i < 2 * kNumThreads / 3; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          if (ch->Receive(&data)) {
            EXPECT_EQ(data, 10);
          }
          *p = true;
        },
        &thread_ended[i]);
  }

  // Launches threads that try to close the channel.
  for (size_t i = 2 * kNumThreads / 3; i < kNumThreads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          if (!ch->IsClosed()) {
            ch->close();
          }
          *p = true;
        },
        &thread_ended[i]);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait

  // Verify that all threads are unblocked
  for (size_t i = 0; i < kNumThreads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }
  EXPECT_TRUE(ch->IsClosed());
  // delete the channel
  delete ch;
  for (size_t i = 0; i < kNumThreads; i++) t[i].join();
}

TEST(ChannelHolder, ChannelHolderManyTimesCloseTest) {
  // Check for Buffered Channel
  ChannelHolder *ch = new ChannelHolder();
  ch->Reset<int>(10);
  ChannelHolderManyTimesClose(ch);
}
