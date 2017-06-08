#include "paddle/majel/place.h"
#include <sstream>
#include "gtest/gtest.h"

TEST(Place, Equality) {
  majel::CpuPlace cpu;
  EXPECT_EQ(cpu, cpu);

#ifndef PADDLE_ONLY_CPU
  majel::GpuPlace g0(0), g1(1), gg0(0);
  EXPECT_EQ(g0, g0);
  EXPECT_EQ(g1, g1);
  EXPECT_EQ(g0, gg0);
  EXPECT_NE(g0, g1);
  EXPECT_TRUE(majel::places_are_same_class(g0, gg0));
  EXPECT_FALSE(majel::places_are_same_class(g0, cpu));
#endif
}

TEST(Place, Default) {
#ifndef PADDLE_ONLY_CPU
  EXPECT_TRUE(majel::is_gpu_place(majel::get_place()));
  EXPECT_TRUE(majel::is_gpu_place(majel::default_gpu()));
  EXPECT_TRUE(majel::is_cpu_place(majel::default_cpu()));
#endif
  majel::set_place(majel::CpuPlace());
  EXPECT_TRUE(majel::is_cpu_place(majel::get_place()));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << majel::CpuPlace();
    EXPECT_EQ("CpuPlace", ss.str());
  }

#ifndef PADDLE_ONLY_CPU
  {
    std::stringstream ss;
    ss << majel::GpuPlace(1);
    EXPECT_EQ("GpuPlace(1)", ss.str());
  }
#endif
}
