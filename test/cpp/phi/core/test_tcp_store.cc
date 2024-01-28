// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace phi {
namespace distributed {

TEST(MasterDaemon, init) {
  int socket = tcputils::tcp_listen("", std::to_string(0), AF_INET);
  auto d = detail::MasterDaemon::start(socket, 1, 100);
  printf("started to sleep 2s\n");
#ifdef _WIN32
  Sleep(2 * 1000);
#else
  usleep(2 * 1000 * 1000);
#endif
  printf("end to reset\n");

  d.reset();
}

/* now for only c compile test
TEST(TCPStore, init) {
  TCPStore store("127.0.0.1", 6170, true, 1);
  store.add("my", 3);
  auto ret1 = store.get("my");
  store.add("my", 3);
  auto ret2 = store.get("my");
  PADDLE_ENFORCE_EQ(ret1[0] + 3, ret2[0],
    paddle::errors::Fatal("result of add is not right"));
}
*/
}  // namespace distributed
}  // namespace phi
