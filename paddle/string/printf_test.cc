#include "paddle/string/printf.h"

#include <string>

#include "gtest/gtest.h"

TEST(StringPrintf, StringPrintf) {
  std::string weekday = "Wednesday";
  const char* month = "July";
  size_t day = 27;
  long hour = 14;
  int min = 44;
  EXPECT_EQ(std::string("Wednesday, July 27, 14:44"),
            paddle::string::Sprintf("%s, %s %d, %.2d:%.2d", weekday, month, day,
                                    hour, min));
}
