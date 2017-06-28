#include <gtest/gtest.h>
#include <paddle/platform/enforce.h>

TEST(ENFORCE, OK) {
  PADDLE_ENFORCE(true, "Enforce is ok", 123, "now", 0.345);
  size_t val = 1;
  const size_t limit = 10;
  PADDLE_ENFORCE(val < limit, "Enforce is OK too");
}

TEST(ENFORCE, FAILED) {
  bool in_catch = false;
  try {
    PADDLE_ENFORCE(false, "Enforce is not ok ", 123, " at all");
  } catch (paddle::platform::EnforceNotMet err) {
    in_catch = true;
    std::string msg = "Enforce is not ok 123 at all";
    const char* what = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}