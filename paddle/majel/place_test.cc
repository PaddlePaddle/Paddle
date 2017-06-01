#include "paddle/majel/place.h"
#include <sstream>
#include "gtest/gtest.h"
#include "paddle/utils/Logging.h"

TEST(Place, Equality) {
  majel::CpuPlace cpu;
  EXPECT_EQ(cpu, cpu);

#ifndef PADDLE_ONLY_CPU
  majel::GpuPlace gpu;
  EXPECT_EQ(gpu, gpu);
  EXPECT_FALSE(majel::places_are_same_class(cpu, gpu));
#endif
}

TEST(Place, Default) { EXPECT_TRUE(majel::is_cpu_place(majel::get_place())); }

TEST(Place, Print) {
#ifndef PADDLE_ONLY_CPU
  {
    std::stringstream ss;
    ss << majel::GpuPlace();
    EXPECT_EQ("GpuPlace", ss.str());
  }
#endif
  {
    std::stringstream ss;
    ss << majel::CpuPlace();
    EXPECT_EQ("CpuPlace", ss.str());
  }
  LOG(INFO) << "\n[----------] Done \n";
}
