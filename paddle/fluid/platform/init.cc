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
#include <csignal>
#include <fstream>
#include <string>

#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/string/split.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cupti.h"
#endif
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/place.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

#ifdef WITH_WIN_DUMP_DBG
#include <stdio.h>
#include <time.h>
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>

#include "DbgHelp.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/custom_kernel.h"

DECLARE_int32(paddle_num_threads);
PADDLE_DEFINE_EXPORTED_int32(
    multiple_of_cupti_buffer_size,
    1,
    "Multiple of the CUPTI device buffer size. If the timestamps have "
    "been dropped when you are profiling, try increasing this value.");

namespace paddle {
namespace framework {

#ifdef _WIN32
#define strdup _strdup
#endif

std::once_flag gflags_init_flag;
std::once_flag glog_init_flag;
std::once_flag memory_method_init_flag;

bool InitGflags(std::vector<std::string> args) {
  bool successed = false;
  std::call_once(gflags_init_flag, [&]() {
    FLAGS_logtostderr = true;
    // NOTE(zhiqiu): dummy is needed, since the function
    // ParseNewCommandLineFlags in gflags.cc starts processing
    // commandline strings from idx 1.
    // The reason is, it assumes that the first one (idx 0) is
    // the filename of executable file.
    args.insert(args.begin(), "dummy");
    std::vector<char *> argv;
    std::string line;
    int argc = args.size();
    for (auto &arg : args) {
      argv.push_back(const_cast<char *>(arg.data()));
      line += arg;
      line += ' ';
    }
    VLOG(1) << "Before Parse: argc is " << argc
            << ", Init commandline: " << line;

    char **arr = argv.data();
    ::GFLAGS_NAMESPACE::AllowCommandLineReparsing();
    ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &arr, true);
    successed = true;

    VLOG(1) << "After Parse: argc is " << argc;
  });
  return successed;
}

#ifdef PADDLE_WITH_CUDA
void InitCupti() {
#ifdef PADDLE_WITH_CUPTI
  if (FLAGS_multiple_of_cupti_buffer_size == 1) return;
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
#define MULTIPLY_ATTR_VALUE(attr)                                      \
  {                                                                    \
    PADDLE_ENFORCE_EQ(                                                 \
        !platform::dynload::cuptiActivityGetAttribute(                 \
            attr, &attrValueSize, &attrValue),                         \
        true,                                                          \
        platform::errors::Unavailable("Get cupti attribute failed.")); \
    attrValue *= FLAGS_multiple_of_cupti_buffer_size;                  \
    LOG(WARNING) << "Set " #attr " " << attrValue << " byte";          \
    PADDLE_ENFORCE_EQ(                                                 \
        !platform::dynload::cuptiActivitySetAttribute(                 \
            attr, &attrValueSize, &attrValue),                         \
        true,                                                          \
        platform::errors::Unavailable("Set cupti attribute failed.")); \
  }
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE);
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP);
#if CUDA_VERSION >= 9000
  MULTIPLY_ATTR_VALUE(CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE);
#endif
#undef MULTIPLY_ATTR_VALUE
#endif
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void LoadCustomDevice(const std::string &library_dir) {
  LOG(INFO) << "Try loading custom device libs from: [" << library_dir << "]";
  std::vector<std::string> libs = phi::ListAllLibraries(library_dir);
  for (const auto &lib_path : libs) {
    auto dso_handle = dlopen(lib_path.c_str(), RTLD_NOW);
    PADDLE_ENFORCE_NOT_NULL(
        dso_handle,
        platform::errors::InvalidArgument(
            "Fail to open library: %s with error: %s", lib_path, dlerror()));

    phi::LoadCustomRuntimeLib(lib_path, dso_handle);
  }
  phi::CustomKernelMap::Instance().RegisterCustomKernels();
  LOG(INFO) << "Finished in LoadCustomDevice with libs_path: [" << library_dir
            << "]";
}
#endif

static std::once_flag init_devices_flag;

void InitDevices() {
  std::call_once(init_devices_flag, []() {
    // set name at the entry point of Paddle
    platform::SetCurrentThreadName("MainThread");
// CUPTI attribute should be set before any CUDA context is created (see CUPTI
// documentation about CUpti_ActivityAttribute).
#ifdef PADDLE_WITH_CUDA
    InitCupti();
#endif
    /*Init all available devices by default */
    std::vector<int> devices;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    try {
      // use user specified GPUs in single-node multi-process mode.
      devices = platform::GetSelectedDevices();
    } catch (const std::exception &exp) {
      LOG(WARNING) << "Compiled with WITH_GPU, but no GPU found in runtime.";
    }
#endif
#ifdef PADDLE_WITH_XPU
    try {
      // use user specified XPUs in single-node multi-process mode.
      devices = platform::GetXPUSelectedDevices();
    } catch (const std::exception &exp) {
      LOG(WARNING) << "Compiled with WITH_XPU, but no XPU found in runtime.";
    }
#endif
#ifdef PADDLE_WITH_IPU
    try {
      // use user specified IPUs.
      devices = platform::GetSelectedIPUDevices();
    } catch (const std::exception &exp) {
      LOG(WARNING)
          << "Compiled with PADDLE_WITH_IPU, but no IPU found in runtime.";
    }
#endif
    InitDevices(devices);
  });
}

void InitDevices(const std::vector<int> devices) {
  std::vector<platform::Place> places;

  for (size_t i = 0; i < devices.size(); ++i) {
    // In multi process multi gpu mode, we may have gpuid = 7
    // but count = 1.
    if (devices[i] < 0) {
      LOG(WARNING) << "Invalid devices id.";
      continue;
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    places.emplace_back(platform::CUDAPlace(devices[i]));
#endif
#ifdef PADDLE_WITH_XPU
    places.emplace_back(platform::XPUPlace(devices[i]));
#endif
#ifdef PADDLE_WITH_IPU
    places.emplace_back(platform::IPUPlace(devices[i]));
#endif
#ifdef PADDLE_WITH_ASCEND_CL
    places.emplace_back(platform::NPUPlace(devices[i]));
#endif
  }
  places.emplace_back(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  places.emplace_back(platform::CUDAPinnedPlace());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  const char *custom_kernel_root_p = std::getenv("CUSTOM_DEVICE_ROOT");
  if (!custom_kernel_root_p) {
    VLOG(3) << "Env [CUSTOM_DEVICE_ROOT] is not set.";
  } else {
    std::string custom_kernel_root(custom_kernel_root_p);
    if (!custom_kernel_root.empty()) {
      LOG(INFO) << "ENV [CUSTOM_DEVICE_ROOT]=" << custom_kernel_root;
      LoadCustomDevice(custom_kernel_root);

      auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
      for (auto &dev_type : device_types) {
        auto device_list = phi::DeviceManager::GetSelectedDeviceList(dev_type);
        LOG(INFO) << "CustomDevice: " << dev_type
                  << ", visible devices count: " << device_list.size();
        for (auto &dev_id : device_list) {
          places.push_back(platform::CustomPlace(dev_type, dev_id));
        }
      }
    } else {
      VLOG(3) << "ENV [CUSTOM_DEVICE_ROOT] is empty.";
    }
  }
#endif
  platform::DeviceContextPool::Init(places);

#ifndef PADDLE_WITH_MKLDNN
  platform::SetNumThreads(FLAGS_paddle_num_threads);
#endif
}

#ifndef _WIN32
// Description Quoted from
// https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html
const struct {
  int signal_number;
  const char *name;
  const char *error_string;
} SignalErrorStrings[] = {
    {SIGSEGV, "SIGSEGV", "Segmentation fault"},
    {SIGILL, "SIGILL", "Illegal instruction"},
    {SIGFPE, "SIGFPE", "Erroneous arithmetic operation"},
    {SIGABRT, "SIGABRT", "Process abort signal"},
    {SIGBUS, "SIGBUS", "Access to an undefined portion of a memory object"},
    {SIGTERM, "SIGTERM", "Termination signal"},
};

bool StartsWith(const char *str, const char *prefix) {
  size_t len_prefix = strlen(prefix);
  size_t len_str = strlen(str);
  return len_str < len_prefix ? false : memcmp(prefix, str, len_prefix) == 0;
}

const char *ParseSignalErrorString(const std::string &str) {
  for (size_t i = 0;
       i < (sizeof(SignalErrorStrings) / sizeof(*(SignalErrorStrings)));
       ++i) {
    if (std::string::npos != str.find(SignalErrorStrings[i].name)) {
      return SignalErrorStrings[i].error_string;
    }
  }
  return "Unknown signal";
}

// Handle SIGSEGV, SIGILL, SIGFPE, SIGABRT, SIGBUS, and SIGTERM.
void SignalHandle(const char *data, int size) {
  try {
    // NOTE1: The glog FailureSignalHandler dumped messages
    //   are deal with line by line
    auto signal_msg_dunmer_ptr = SignalMessageDumper::Instance().Get();
    // NOTE2: we only deal with the time info ane signal info,
    //   the stack trace will generated by paddle self
    if (StartsWith(data, "*** Aborted at")) {
      *signal_msg_dunmer_ptr << "\n  [TimeInfo: " << std::string(data, size - 1)
                             << "]\n";
    } else if (StartsWith(data, "***")) {
      std::string signal_info(data, size - 1);
      std::string useless_substr("; stack trace:");
      size_t start_pos = signal_info.rfind(useless_substr);
      signal_info.replace(start_pos, useless_substr.length(), "");
      *signal_msg_dunmer_ptr << "  [SignalInfo: " << signal_info << "]\n";

      // NOTE3: Final singal error message print.
      // Here does not throw an exception,
      // otherwise it will casue "terminate called recursively"
      std::ostringstream sout;
      sout << "\n\n--------------------------------------\n";
      sout << "C++ Traceback (most recent call last):";
      sout << "\n--------------------------------------\n";
      auto traceback = platform::GetCurrentTraceBackString(/*for_signal=*/true);
      if (traceback.empty()) {
        sout
            << "No stack trace in paddle, may be caused by external reasons.\n";
      } else {
        sout << traceback;
      }

      sout << "\n----------------------\nError Message "
              "Summary:\n----------------------\n";
      sout << platform::errors::Fatal(
                  "`%s` is detected by the operating system.",
                  ParseSignalErrorString(signal_info))
                  .to_string();
      std::cout << sout.str() << (*signal_msg_dunmer_ptr).str() << std::endl;
    }
  } catch (...) {
    // Since the program has already triggered a system error,
    // no further processing is required here, glog FailureSignalHandler
    // will Kill program by the default signal handler
  }
}
#endif  // _WIN32

void DisableSignalHandler() {
#ifndef _WIN32
  for (size_t i = 0;
       i < (sizeof(SignalErrorStrings) / sizeof(*(SignalErrorStrings)));
       ++i) {
    int signal_number = SignalErrorStrings[i].signal_number;
    struct sigaction sig_action;
    memset(&sig_action, 0, sizeof(sig_action));
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_handler = SIG_DFL;
    sigaction(signal_number, &sig_action, NULL);
  }
#endif
}

#ifdef WITH_WIN_DUMP_DBG
typedef BOOL(WINAPI *MINIDUMP_WRITE_DUMP)(
    IN HANDLE hProcess,
    IN DWORD ProcessId,
    IN HANDLE hFile,
    IN MINIDUMP_TYPE DumpType,
    IN CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
    OPTIONAL IN PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
    OPTIONAL IN PMINIDUMP_CALLBACK_INFORMATION CallbackParam OPTIONAL);
void CreateDumpFile(LPCSTR lpstrDumpFilePathName,
                    EXCEPTION_POINTERS *pException) {
  HANDLE hDumpFile = CreateFile(lpstrDumpFilePathName,
                                GENERIC_WRITE,
                                0,
                                NULL,
                                CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                NULL);
  MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
  dumpInfo.ExceptionPointers = pException;
  dumpInfo.ThreadId = GetCurrentThreadId();
  dumpInfo.ClientPointers = TRUE;
  MINIDUMP_WRITE_DUMP MiniDumpWriteDump_;
  HMODULE hDbgHelp = LoadLibrary("DBGHELP.DLL");
  MiniDumpWriteDump_ =
      (MINIDUMP_WRITE_DUMP)GetProcAddress(hDbgHelp, "MiniDumpWriteDump");
  MiniDumpWriteDump_(GetCurrentProcess(),
                     GetCurrentProcessId(),
                     hDumpFile,
                     MiniDumpWithPrivateReadWriteMemory,
                     &dumpInfo,
                     NULL,
                     NULL);
  CloseHandle(hDumpFile);
}

LONG ApplicationCrashHandler(EXCEPTION_POINTERS *pException) {
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);

  char buf[1024];
  sprintf_s(buf,
            "C:\\Paddle%04d%02d%02d-%02d%02d%02d.dmp",
            1900 + now_time.tm_year,
            1 + now_time.tm_mon,
            now_time.tm_mday,
            now_time.tm_hour,
            now_time.tm_min,
            now_time.tm_sec);

  CreateDumpFile(buf, pException);
  return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void InitGLOG(const std::string &prog_name) {
  std::call_once(glog_init_flag, [&]() {
// glog will not hold the ARGV[0] inside.
// Use strdup to alloc a new string.
#ifdef WITH_WIN_DUMP_DBG
    SetUnhandledExceptionFilter(
        (LPTOP_LEVEL_EXCEPTION_FILTER)ApplicationCrashHandler);
#endif
    google::InitGoogleLogging(strdup(prog_name.c_str()));
#ifndef _WIN32
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter(&SignalHandle);
#endif
  });
}

void InitMemoryMethod() {
  std::call_once(memory_method_init_flag, [&]() {
    auto &memory_utils = phi::MemoryUtils::Instance();
    auto memory_method = std::make_unique<phi::MemoryInterface>();
    memory_method->alloc = paddle::memory::Alloc;
    memory_method->alloc_with_stream = paddle::memory::Alloc;
    memory_method->alloc_shared = paddle::memory::AllocShared;
    memory_method->alloc_shared_with_stream = paddle::memory::AllocShared;
    memory_method->in_same_stream = paddle::memory::InSameStream;
    memory_method->allocation_deleter =
        paddle::memory::allocation::Allocator::AllocationDeleter;
#if defined(PADDLE_WITH_CUSTOM_DEVICE) || defined(PADDLE_WITH_CUDA) || \
    defined(PADDLE_WITH_HIP)
    memory_method->copy_with_stream =
        paddle::memory::Copy<phi::Place, phi::Place>;
#endif
    memory_method->copy = paddle::memory::Copy<phi::Place, phi::Place>;
    memory_method->device_memory_stat_current_value =
        paddle::memory::DeviceMemoryStatCurrentValue;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    memory_method->gpu_memory_usage = paddle::platform::GpuMemoryUsage;
#endif
    memory_method->emplace_device_contexts =
        paddle::platform::EmplaceDeviceContexts;
    memory_method->init_devices = InitDevices;
    memory_utils.Init(std::move(memory_method));
  });
}

}  // namespace framework
}  // namespace paddle
