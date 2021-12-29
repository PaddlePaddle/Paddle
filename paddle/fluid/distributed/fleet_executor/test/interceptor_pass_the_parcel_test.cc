// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/global_map.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"

namespace paddle {
namespace distributed {

class ParcelInterceptor : public Interceptor {
 public:
  ParcelInterceptor(int64_t interceptor_id, TaskNode* node)
      : Interceptor(interceptor_id, node) {
    RegisterMsgHandle(
        [this](const InterceptorMessage& msg) { PassParcel(msg); });
  }

  void PassParcel(const InterceptorMessage& msg) {
    if (msg.message_type() == STOP) {
      stop_ = true;
      return;
    }
    std::cout << GetInterceptorId() << " recv msg, count=" << count_
              << std::endl;
    if (count_ == 5 && interceptor_id_ == 0) {
      InterceptorMessage stop;
      stop.set_message_type(STOP);
      Send(0, stop);
      Send(1, stop);
      Send(2, stop);
      Send(3, stop);
      StopCarrier();
      return;
    }
    ++count_;
    InterceptorMessage new_msg;
    if (msg.dst_id() == 3) {
      Send(0, new_msg);
    } else {
      Send(msg.dst_id() + 1, new_msg);
    }
  }

 private:
  int count_{0};
};

REGISTER_INTERCEPTOR(Parcel, ParcelInterceptor);

TEST(InterceptorTest, PassTheParcel) {
  auto msg_bus = std::make_shared<MessageBus>();
  Carrier* carrier_0 = GlobalMap<int64_t, Carrier>::Create(0, 0);
  carrier_0->Init(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {0});
  carrier_0->SetMsgBus(msg_bus);
  Carrier* carrier_1 = GlobalMap<int64_t, Carrier>::Create(1, 1);
  carrier_1->Init(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {1});
  carrier_1->SetMsgBus(msg_bus);
  Carrier* carrier_2 = GlobalMap<int64_t, Carrier>::Create(2, 2);
  carrier_2->Init(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {2});
  carrier_2->SetMsgBus(msg_bus);
  Carrier* carrier_3 = GlobalMap<int64_t, Carrier>::Create(3, 3);
  carrier_3->Init(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {3});
  carrier_3->SetMsgBus(msg_bus);
  msg_bus->Init(0, {{0, "127.0.0.0:0"}}, "");

  Interceptor* a = carrier_0->SetInterceptor(
      0, InterceptorFactory::Create("Parcel", 0, nullptr));

  carrier_1->SetInterceptor(1,
                            InterceptorFactory::Create("Parcel", 1, nullptr));
  carrier_2->SetInterceptor(2,
                            InterceptorFactory::Create("Parcel", 2, nullptr));
  carrier_3->SetInterceptor(3,
                            InterceptorFactory::Create("Parcel", 3, nullptr));

  InterceptorMessage msg;
  a->Send(1, msg);

  carrier_0->Wait();
}

}  // namespace distributed
}  // namespace paddle
