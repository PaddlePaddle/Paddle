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

#include "paddle/fluid/imperative/data_loader.h"

#include <cstdlib>

#include <csignal>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"

#ifdef _WIN32
#include <processthreadsapi.h>
#include <tlhelp32.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace paddle::imperative {

static std::map<int64_t, std::set<pid_t>> load_process_pids;

void SetLoadProcessPIDs(int64_t key, std::set<pid_t> pids) {
  VLOG(3) << "DataLoader: set loader child process PID (" << key
          << ", pid number: " << pids.size() << ")";
  load_process_pids[key] = pids;
}

void EraseLoadProcessPIDs(int64_t key) {
  auto it = load_process_pids.find(key);
  // Note: Can not find key also possible
  if (it != load_process_pids.end()) {
    VLOG(3) << "Dygraph Data Loader: erase loader child process PID (" << key
            << ")";
    load_process_pids.erase(it);
  } else {
    VLOG(3) << "Dygraph Data Loader: The dygraph loader (id: " << key
            << ") you want erase does not exist.";
  }
}

#ifndef _WIN32
// sigaction doc: http://man7.org/linux/man-pages/man2/sigaction.2.html
// sigemptyset doc: https://linux.die.net/man/3/sigemptyset
// siginfo_t doc: https://www.mkssoftware.com/docs/man5/siginfo_t.5.asp
// waitid doc: https://linux.die.net/man/2/waitid

// clear mmap fds on signal handler, make sure mmap clear will be called
// on signal handling and no need to register mmap clear up handler on
// python side. If shared memory is not used Clear() will do nothing.
#define SIGNAL_HANDLE(SIGNAL)                               \
  do {                                                      \
    memory::allocation::MemoryMapFdSet::Instance().Clear(); \
    struct sigaction sa = {};                               \
    sa.sa_handler = SIG_DFL;                                \
    sa.sa_flags = 0;                                        \
    if (sigemptyset(&sa.sa_mask) != 0 ||                    \
        sigaction(SIGNAL, &sa, nullptr) != 0) {             \
      _exit(EXIT_FAILURE);                                  \
    } else {                                                \
      raise(SIGNAL);                                        \
    }                                                       \
  } while (0)

#define REGISTER_SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)           \
  static void HANDLER_NAME(                                                \
      int sig UNUSED, siginfo_t *info UNUSED, void *ctx UNUSED) {          \
    auto _w =                                                              \
        write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char)); \
    (void)_w;                                                              \
    SIGNAL_HANDLE(SIGNAL);                                                 \
  }

#define REGISTER_SPEC_SIGNAL_HANDLER(SIGNAL, HANDLER_NAME)        \
  static void HANDLER_NAME(int sig, siginfo_t *info, void *ctx) { \
    if (info->si_pid == getppid()) {                              \
      _exit(EXIT_SUCCESS);                                        \
    }                                                             \
    SIGNAL_HANDLE(SIGNAL);                                        \
  }

REGISTER_SIGNAL_HANDLER(SIGSEGV,
                        SIGSEGV_handler,
                        "ERROR: Unexpected segmentation fault encountered in "
                        "DataLoader workers.\n");
REGISTER_SIGNAL_HANDLER(
    SIGBUS,
    SIGBUS_handler,
    "ERROR: Unexpected BUS error encountered in DataLoader worker. "
    "This might be caused by insufficient shared memory (shm), "
    "please check whether use_shared_memory is set and storage space "
    "in /dev/shm is enough\n");
REGISTER_SIGNAL_HANDLER(SIGFPE,
                        SIGFPE_handler,
                        "ERROR: Unexpected floating-point exception "
                        "encountered in DataLoader worker.\n")
REGISTER_SPEC_SIGNAL_HANDLER(SIGTERM, SIGTERM_handler);

static inline void setSignalHandler(int signal,
                                    void (*handler)(int, siginfo_t *, void *),
                                    struct sigaction *old_sa_ptr) {
  struct sigaction sa;
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 ||
      sigaction(signal, &sa, old_sa_ptr) != 0) {
    PADDLE_THROW(common::errors::Fatal(
        "An error occurred while setting handler for %s.", strsignal(signal)));
  }
}
#endif

#ifdef _WIN32
// Last-chance exception filter: logs crash info before OS terminates the
// process.
static LONG CALLBACK paddle_crash_filter(EXCEPTION_POINTERS *ep) {
  fprintf(stderr,
          "[PADDLE_CRASH] PID=%lu code=0x%08lX addr=%p\n",
          GetCurrentProcessId(),
          ep->ExceptionRecord->ExceptionCode,
          ep->ExceptionRecord->ExceptionAddress);
  fflush(stderr);
  return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void SetLoadProcessSignalHandler() {
#ifdef _WIN32
  // Last-chance filter: logs crash details before OS terminates the process.
  // This is the ONLY handler on Windows — VEH is removed because it could
  // deadlock if MemoryMapFdSet::Clear() is called while the mutex is held.
  SetUnhandledExceptionFilter(&paddle_crash_filter);
  VLOG(3) << "DataLoader: crash handler registered";
#else
  setSignalHandler(SIGSEGV, &SIGSEGV_handler, nullptr);
  setSignalHandler(SIGBUS, &SIGBUS_handler, nullptr);
  setSignalHandler(SIGFPE, &SIGFPE_handler, nullptr);
  setSignalHandler(SIGTERM, &SIGTERM_handler, nullptr);
#endif
}

void ThrowErrorIfLoadProcessFailed() {
#ifdef _WIN32
  // On Windows, use WaitForSingleObject + GetExitCodeProcess
  // to check for crashed workers. This is called periodically from Python.
  int error = 0;
  std::set<pid_t> *pids_set = nullptr;
  pid_t process_pid = 0;

  for (auto &p : load_process_pids) {
    pids_set = &(p.second);
    for (auto pid_it = pids_set->begin(); pid_it != pids_set->end(); ++pid_it) {
      process_pid = *pid_it;
      VLOG(3) << "DataLoader: monitor loader child process " << process_pid;

      HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | SYNCHRONIZE,
                                    FALSE,
                                    static_cast<DWORD>(process_pid));
      if (hProcess == NULL) {
        // Process may have already exited; GetLastError() ==
        // ERROR_INVALID_PARAMETER
        continue;
      }

      DWORD exit_code = STILL_ACTIVE;
      if (WaitForSingleObject(hProcess, 0) == WAIT_OBJECT_0) {
        // Process has exited
        GetExitCodeProcess(hProcess, &exit_code);
        CloseHandle(hProcess);

        if (exit_code != 0 && exit_code != STILL_ACTIVE) {
          // Exited with error (non-zero exit code)
          fprintf(stderr,
                  "[DL_WORKER_EXIT] pid=%lu exit_code=0x%08lX (%lu)\n",
                  process_pid,
                  exit_code,
                  exit_code);
          fflush(stderr);
          pids_set->clear();
          PADDLE_THROW(common::errors::Fatal(
              "DataLoader process (pid %ld) exited unexpectedly with code "
              "0x%lX (%lu). "
              "Rerunning with num_workers=0 may give better error trace.",
              process_pid,
              exit_code,
              exit_code));
        }
      } else {
        CloseHandle(hProcess);
      }
    }
  }
#else
  int error = 0;
  std::set<pid_t> *pids_set = nullptr;
  pid_t process_pid = 0;
  siginfo_t infop;

  for (auto &p : load_process_pids) {
    pids_set = &(p.second);
    for (auto pid_it = pids_set->begin(); pid_it != pids_set->end(); ++pid_it) {
      process_pid = *pid_it;
      // Use waitid rather than waitpid so that we can set NOWAIT, and that
      // Python and other handlers can get whatever info they want about the
      // child.
      infop.si_pid = 0;
      VLOG(3) << "DataLoader: monitor loader child process " << process_pid;
      error = waitid(P_PID, process_pid, &infop, WEXITED | WNOHANG | WNOWAIT);
      // ignore errors and case with no waitable child
      if (error < 0 || infop.si_pid == 0) continue;
      if (infop.si_code == CLD_EXITED &&
          infop.si_status != EXIT_SUCCESS) {  // exit with error
        pids_set->clear();
        PADDLE_THROW(common::errors::Fatal(
            "DataLoader process (pid %ld) exited unexpectedly with code %d. "
            "Error detailed are lost due to multiprocessing. Rerunning with:\n"
            "  1. If run DataLoader by DataLoader.from_generator(...), run "
            "with "
            "DataLoader.from_generator(..., use_multiprocess=False) may give "
            "better error trace.\n"
            "  2. If run DataLoader by DataLoader(dataset, ...), run with "
            "DataLoader(dataset, ..., num_workers=0) may give better error "
            "trace",
            process_pid,
            infop.si_status));
      } else if (infop.si_code == CLD_KILLED ||
                 infop.si_code == CLD_DUMPED) {  // killed by signal
        if (infop.si_status == SIGBUS) {
          pids_set->clear();
          PADDLE_THROW(common::errors::Fatal(
              "DataLoader process (pid %ld) exited is killed by signal: %s.\n"
              "  It may be caused by insufficient shared storage space. This "
              "problem usually occurs when using docker as a development "
              "environment.\n  Please use command `df -h` to check the storage "
              "space of `/dev/shm`. Shared storage space needs to be greater "
              "than (DataLoader Num * DataLoader queue capacity * 1 batch data "
              "size).\n  You can solve this problem by increasing the shared "
              "storage space or reducing the queue capacity appropriately.\n"
              "  1. If run DataLoader by DataLoader.from_generator(...), queue "
              "capacity is set by from_generator(..., capacity=xx, ...).\n"
              "  2. If run DataLoader by DataLoader(dataset, ...), queue "
              "capacity is set as 2 times of the max value of num_workers and "
              "len(places).\n"
              "  3. If run by DataLoader(dataset, ..., use_shared_memory=True),"
              " set use_shared_memory=False for not using shared memory.",
              process_pid,
              strsignal(infop.si_status)));
        } else {
          PADDLE_THROW(common::errors::Fatal(
              "DataLoader process (pid %ld) exited is killed by signal: %s.",
              process_pid,
              strsignal(infop.si_status)));
        }
      }
    }
  }
#endif
}

}  // namespace paddle::imperative
