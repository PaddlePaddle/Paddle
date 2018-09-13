/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

using paddle::operators::distributed::VarHandlePtr;
using paddle::operators::distributed::VarHandle;

void WaitTrue(VarHandlePtr s) { EXPECT_TRUE(s->Wait()); }

void WaitFalse(VarHandlePtr s) { EXPECT_FALSE(s->Wait()); }

TEST(VarHandle, Run) {
  std::vector<VarHandlePtr> a;
  for (int i = 0; i < 12; i++) {
    VarHandlePtr s(new VarHandle("", "", "", nullptr, nullptr));
    a.push_back(s);
  }

  std::vector<std::unique_ptr<std::thread>> t;
  for (int i = 0; i < 6; i++) {
    t.emplace_back(new std::thread(WaitFalse, a[i]));
  }

  for (int i = 0; i < 6; i++) {
    a[i]->Finish(false);
    t[i]->join();
  }

  for (int i = 6; i < 12; i++) {
    t.emplace_back(new std::thread(WaitTrue, a[i]));
  }

  for (int i = 6; i < 12; i++) {
    a[i]->Finish(true);
    t[i]->join();
  }
}
