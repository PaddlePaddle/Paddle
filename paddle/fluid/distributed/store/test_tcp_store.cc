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
#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/distributed/store/tcp_utils.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace paddle {
namespace distributed {

int get_free_port() {
  unsigned int seed = time(0);
  int port = 10000 + rand_r(&seed) % 3000;
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  linger ling;
  ling.l_onoff = 1;
  ling.l_linger = 0;
  setsockopt(server_fd, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);
  while (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == -1) {
    port++;
    address.sin_port = htons(port);
  }
  close(server_fd);
  return port;
}

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

TEST(TCPStore, set_after_get) {
  int rank = fork();
  bool is_master = (rank == 0);
  int port = get_free_port();
  TCPStore store("127.0.0.1", port, is_master, 2, 10);
  if (rank == 0) {
#ifdef _WIN32
    Sleep(2 * 1000);
#else
    usleep(2 * 1000 * 1000);
#endif
    store.set("bar", {3});
  } else {
    auto res = store.get("bar");
  }
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

};  // namespace distributed
};  // namespace paddle
