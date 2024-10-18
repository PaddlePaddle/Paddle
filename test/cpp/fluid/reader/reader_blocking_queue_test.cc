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

#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/phi/core/operators/reader/blocking_queue.h"

using paddle::operators::reader::BlockingQueue;

TEST(BlockingQueue, CapacityTest) {
  size_t cap = 10;
  BlockingQueue<int> q(cap);
  EXPECT_EQ(q.Cap(), cap);
}

void FirstInFirstOut(size_t queue_cap,
                     size_t elem_num,
                     size_t send_time_gap,
                     size_t receive_time_gap) {
  BlockingQueue<size_t> q(queue_cap);
  std::thread sender([&]() {
    for (size_t i = 0; i < elem_num; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(send_time_gap));
      EXPECT_TRUE(q.Send(i));
    }
    q.Close();
  });
  size_t count = 0;
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(receive_time_gap));
    size_t elem = 0;
    if (!q.Receive(&elem)) {
      break;
    }
    EXPECT_EQ(elem, count++);
  }
  sender.join();
  EXPECT_EQ(count, elem_num);
  EXPECT_TRUE(q.IsClosed());
}

TEST(BlockingQueue, FirstInFirstOutTest) {
  FirstInFirstOut(2, 5, 2, 50);
  FirstInFirstOut(2, 5, 50, 2);
  FirstInFirstOut(10, 3, 50, 2);
  FirstInFirstOut(10, 3, 2, 50);
}

TEST(BlockingQueue, SenderBlockingTest) {
  const size_t queue_cap = 2;
  BlockingQueue<size_t> q(queue_cap);
  size_t send_count = 0;
  std::thread sender([&]() {
    for (size_t i = 0; i < 5; ++i) {
      if (!q.Send(i)) {
        break;
      }
      ++send_count;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  q.Close();
  sender.join();
  EXPECT_EQ(send_count, queue_cap);
  std::vector<size_t> res;
  while (true) {
    size_t elem = 0;
    if (!q.Receive(&elem)) {
      break;
    }
    res.push_back(elem);
  }
  EXPECT_EQ(res.size(), queue_cap);
  for (size_t i = 0; i < res.size(); ++i) {
    EXPECT_EQ(res[i], i);
  }
}

TEST(BlockingQueue, ReceiverBlockingTest) {
  const size_t queue_cap = 5;
  BlockingQueue<size_t> q(queue_cap);
  std::vector<size_t> receive_res;
  std::thread receiver([&]() {
    size_t elem = 0;
    while (true) {
      if (!q.Receive(&elem)) {
        break;
      }
      receive_res.push_back(elem);
    }
  });
  std::vector<size_t> to_send{2, 1, 7};
  for (auto e : to_send) {
    q.Send(e);
  }
  q.Close();
  receiver.join();
  EXPECT_EQ(receive_res.size(), to_send.size());
  for (size_t i = 0; i < to_send.size(); ++i) {
    EXPECT_EQ(receive_res[i], to_send[i]);
  }
}

void CheckIsUnorderedSame(const std::vector<std::vector<size_t>>& v1,
                          const std::vector<std::vector<size_t>>& v2) {
  std::set<size_t> s1;
  std::set<size_t> s2;
  for (auto const& vec : v1) {
    for (size_t elem : vec) {
      s1.insert(elem);
    }
  }
  for (auto const& vec : v2) {
    for (size_t elem : vec) {
      s2.insert(elem);
    }
  }
  EXPECT_EQ(s1.size(), s2.size());
  auto it1 = s1.begin();
  auto it2 = s2.begin();
  while (it1 != s1.end()) {
    EXPECT_EQ(*it1, *it2);
    ++it1;
    ++it2;
  }
}

void MultiSenderMultiReceiver(const size_t queue_cap,
                              const std::vector<std::vector<size_t>>& to_send,
                              size_t receiver_num,
                              size_t send_time_gap,
                              size_t receive_time_gap) {
  BlockingQueue<size_t> q(queue_cap);
  size_t sender_num = to_send.size();
  std::vector<std::thread> senders;
  for (size_t s_idx = 0; s_idx < sender_num; ++s_idx) {
    senders.emplace_back([&, s_idx] {
      for (size_t elem : to_send[s_idx]) {
        std::this_thread::sleep_for(std::chrono::milliseconds(send_time_gap));
        EXPECT_TRUE(q.Send(elem));
      }
    });
  }
  std::vector<std::thread> receivers;
  std::mutex mu;
  std::vector<std::vector<size_t>> res;
  for (size_t r_idx = 0; r_idx < receiver_num; ++r_idx) {
    receivers.emplace_back([&] {
      std::vector<size_t> receiver_res;
      while (true) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(receive_time_gap));
        size_t elem = 0;
        if (!q.Receive(&elem)) {
          break;
        }
        receiver_res.push_back(elem);
      }
      std::lock_guard<std::mutex> lock(mu);
      res.push_back(receiver_res);
    });
  }
  for (auto& t : senders) {
    t.join();
  }
  q.Close();
  for (auto& t : receivers) {
    t.join();
  }
  CheckIsUnorderedSame(to_send, res);
}

TEST(BlockingQueue, MultiSenderMultiReaderTest) {
  std::vector<std::vector<size_t>> to_send_1{{2, 3, 4}, {9}, {0, 7, 15, 6}};
  MultiSenderMultiReceiver(2, to_send_1, 2, 0, 0);
  MultiSenderMultiReceiver(10, to_send_1, 2, 0, 0);
  MultiSenderMultiReceiver(2, to_send_1, 20, 0, 0);
  MultiSenderMultiReceiver(2, to_send_1, 2, 50, 0);
  MultiSenderMultiReceiver(2, to_send_1, 2, 0, 50);

  std::vector<std::vector<size_t>> to_send_2{
      {2, 3, 4}, {}, {0, 7, 15, 6, 9, 32}};
  MultiSenderMultiReceiver(2, to_send_2, 3, 0, 0);
  MultiSenderMultiReceiver(20, to_send_2, 3, 0, 0);
  MultiSenderMultiReceiver(2, to_send_2, 30, 0, 0);
  MultiSenderMultiReceiver(2, to_send_2, 3, 50, 0);
  MultiSenderMultiReceiver(2, to_send_2, 3, 0, 50);
}

struct MyClass {
  MyClass() : val_(0) {}
  explicit MyClass(int val) : val_(val) {}
  MyClass(const MyClass& b) { val_ = b.val_; }
  MyClass(MyClass&& b) noexcept { val_ = b.val_; }
  MyClass& operator=(const MyClass& b) {
    if (this != &b) {
      val_ = b.val_;
      return *this;
    }
    return *this;
  }

  int val_;
};

TEST(BlockingQueue, MyClassTest) {
  BlockingQueue<MyClass> q(2);
  MyClass a(200);
  q.Send(a);
  MyClass b;
  q.Receive(&b);
  EXPECT_EQ(a.val_, b.val_);
}

TEST(BlockingQueue, speed_test_mode) {
  size_t queue_size = 10;
  BlockingQueue<size_t> q1(queue_size, false);
  for (size_t i = 0; i < queue_size; ++i) {
    q1.Send(i);
  }
  size_t b = 0;
  for (size_t i = 0; i < queue_size; ++i) {
    q1.Receive(&b);
    EXPECT_EQ(b, i);
  }
  EXPECT_EQ(q1.Size(), 0UL);

  BlockingQueue<size_t> q2(queue_size, true);
  for (size_t i = 0; i < queue_size; ++i) {
    q2.Send(i);
  }
  for (size_t i = 0; i < queue_size; ++i) {
    q2.Receive(&b);
    EXPECT_EQ(b, 0UL);
  }
  EXPECT_EQ(q2.Size(), queue_size);
}
