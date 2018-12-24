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

TEST(TimerUsage, Print) {
  platform::Timer timeline;
  timeline.Start();
  sleep(3);
  timeline.Pause();
  std::cout << paddle::string::Sprintf("passed time %f", timeline.ElapsedSec())
            << std::endl;
  timeline.Reset();
  std::cout << paddle::string::Sprintf("after reset %fs", timeline.ElapsedSec())
            << std::endl;

  std::cout << paddle::string::Sprintf("after reset %fus", timeline.ElapsedUS())
            << std::endl;

  std::cout << paddle::string::Sprintf("after reset %fms", timeline.ElapsedMS())
            << std::endl;
  std::cout << paddle::string::Sprintf("count is %d", timeline.Count())
            << std::endl;
}
