#include <gtest/gtest.h>
#include <paddle/framework/typedefs.h>

template <typename Function>
inline void TestSmartPtrImpl(int val, Function func) {
  auto tmp = func();
  *tmp = val;
  ASSERT_TRUE(tmp);
  ASSERT_EQ(val, *tmp);
}

TEST(Typedefs, SharedPtr) {
  TestSmartPtrImpl(32, paddle::framework::MakeShared<int>);
}

TEST(Typedefs, UniquePtr) {
  TestSmartPtrImpl(64, paddle::framework::MakeUnique<int>);
}

struct TestNonCopy : public paddle::framework::NonCopyable {};

TEST(Typedefs, NonCopyAble) {
  //  following lines will make compiler error.
  //  TestNonCopy a;
  //  TestNonCopy b = a;
}
