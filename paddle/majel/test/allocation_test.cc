#include "majel/allocation.h"
#include "gtest/gtest.h"

TEST(Allocation, malloc) {
  majel::CpuPlace cpu;
  majel::Allocation block(100, cpu);
  EXPECT_EQ(static_cast<size_t>(100), block.size());
  EXPECT_TRUE(majel::places_are_same_class(cpu, block.place()));
  EXPECT_EQ(100, (uint8_t*)block.end() - (uint8_t*)block.ptr());
}
