// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef _WIN32

#include "paddle/fluid/imperative/data_loader.h"

#include <string.h>
#include <sys/wait.h>

#include <atomic>
#include <csignal>
#include <map>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

static std::map<int64_t, pid_t> load_process_pids;

void SetLoadProcessPID(int64_t key, pid_t pid) { load_process_pids[key] = pid; }

void EraseLoadProcessPID(int64_t key) {
  auto it = load_process_pids.find(key);
  if (it != load_process_pids.end()) {
    load_process_pids.erase(it);
  }
}

#define REGISTER_SIGNAL_HANDLER(SIGNAL, HANDLER_NAME)             \
  static void HANDLER_NAME(int sig, siginfo_t *info, void *ctx) { \
    struct sigaction sa {};                                       \
    sa.sa_handler = SIG_DFL;                                      \
    sa.sa_flags = 0;                                              \
    if (sigemptyset(&sa.sa_mask) != 0 ||                          \
        sigaction(SIGNAL, &sa, nullptr) != 0) {                   \
      _exit(EXIT_FAILURE);                                        \
    } else {                                                      \
      raise(SIGNAL);                                              \
    }                                                             \
  }

REGISTER_SIGNAL_HANDLER(SIGBUS, handler_SIGBUS);
REGISTER_SIGNAL_HANDLER(SIGSEGV, handler_SIGSEGV);

static void handler_SIGTERM(int sig, siginfo_t *info, void *ctx) {
  if (info->si_pid == getppid()) {
    _exit(EXIT_SUCCESS);
  }
  struct sigaction sa {};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGTERM, &sa, nullptr) != 0) {
    _exit(EXIT_FAILURE);
  } else {
    raise(SIGTERM);
  }
}

static inline void setSignalHandler(int signal,
                                    void (*handler)(int, siginfo_t *, void *),
                                    struct sigaction *old_sa_ptr) {
  struct sigaction sa {};
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 ||
      sigaction(signal, &sa, old_sa_ptr) != 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "An error occurred while setting handler for %s.", strsignal(signal)));
  }
}

void SetLoadProcessSignalHandler() {
  setSignalHandler(SIGBUS, &handler_SIGBUS, nullptr);
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, nullptr);
  setSignalHandler(SIGTERM, &handler_SIGTERM, nullptr);
}

void ThrowErrorIfLoadProcessFailed() {
  int error;
  pid_t process_pid;
  siginfo_t infop;

  for (auto &w : load_process_pids) {
    process_pid = w.second;
    // Use waitid rather than waitpid so that we can set NOWAIT, and that Python
    // and other handlers can get whatever info they want about the child.
    infop.si_pid = 0;
    error = waitid(P_PID, process_pid, &infop, WEXITED | WNOHANG | WNOWAIT);
    // ignore errors and case with no waitable child
    if (error < 0 || infop.si_pid == 0) continue;
    if (infop.si_code == CLD_EXITED &&
        infop.si_status != EXIT_SUCCESS) {  // exit with error
      PADDLE_THROW(platform::errors::Fatal(
          "DataLoader process (pid %ld) exited unexpectedly with code %d. "
          "Error detailed are lost due to multiprocessing. Rerunning with "
          "DataLoader.from_generator(..., use_multiprocess=False) may give "
          "better error trace.",
          process_pid, infop.si_status));
    } else if (infop.si_code == CLD_KILLED ||
               infop.si_code == CLD_DUMPED) {  // killed by signal
      PADDLE_THROW(platform::errors::Fatal(
          "DataLoader process (pid %ld) exited is killed by signal: %s.",
          process_pid, strsignal(infop.si_status)));
    }
  }
}

}  // namespace imperative
}  // namespace paddle

#endif
