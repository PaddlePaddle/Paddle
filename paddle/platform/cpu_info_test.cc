#include "paddle/platform/cpu_info.h"

#include <ostream>
#include <sstream>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DECLARE_double(fraction_of_cpu_memory_to_use);

TEST(CpuMemoryUsage, Print) {
  std::stringstream ss;
  size_t mem_size = paddle::platform::CpuTotalMemory() / 1024 / 1024 / 1024;
  ss << std::to_string(
            static_cast<size_t>(FLAGS_fraction_of_cpu_memory_to_use * 100))
     << "% of CPU Memory Usage: " << mem_size << " GB";
  std::cout << ss.str();
}
