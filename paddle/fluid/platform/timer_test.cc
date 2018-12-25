//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/platform/timer.h"

TEST(Timer, Reset) {
  platform::Timer timeline;
  timeline.Start();
  sleep(3);
  timeline.Pause();
  timeline.Reset();
  std::cout << paddle::string::Sprintf("after reset %fs", timeline.ElapsedSec())
            << std::endl;
}

TEST(Timer, Start) {
  platform::Timer timeline;
  timeline.Start();
  sleep(3);
  timeline.Pause();
  std::cout << paddle::string::Sprintf("after pause %fs", timeline.ElapsedSec())
            << std::endl;
}

TEST(Timer, Pause) {
  platform::Timer timeline;
  timeline.Start();
  sleep(3);
  timeline.Pause();
  std::cout << paddle::string::Sprintf("after pause %fus", timeline.ElapsedUS())
            << std::endl;
}

TEST(Timer, Resume) {
  platform::Timer timeline;
  timeline.Start();
  sleep(3);
  timeline.Pause();
  timeline.Resume();
  std::cout << paddle::string::Sprintf("after pause %fms", timeline.ElapsedMS())
            << std::endl;
}
