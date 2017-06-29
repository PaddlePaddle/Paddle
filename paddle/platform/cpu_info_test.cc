#include "paddle/platform/cpu_info.h"
#include "paddle/string/printf.h"

#include <ostream>
#include <sstream>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

DECLARE_double(fraction_of_cpu_memory_to_use);

TEST(CpuMemoryUsage, Print) {
  std::stringstream ss;
  size_t memory_size = paddle::platform::CpuMaxAllocSize() / 1024 / 1024 / 1024;
  float use_percent = FLAGS_fraction_of_cpu_memory_to_use * 100;

  std::cout << paddle::string::Sprintf("\n%.2f %% of CPU Memory Usage: %d GB\n",
                                       use_percent, memory_size)
            << std::endl;
}
