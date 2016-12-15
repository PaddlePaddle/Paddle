#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DEFINE_bool(verbose, false, "Display program name before message");
DEFINE_string(message, "Hello world!", "Message to print");

static bool IsNonEmptyMessage(const char *flagname, const std::string &value) {
  return value[0] != '\0';
}
DEFINE_validator(message, &IsNonEmptyMessage);

namespace third_party {
namespace gflags_test {

TEST(GflagsTest, ParseAndPrint) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  int argc = 1;
  char program_name[] = "gflags_test";
  char **argv = new char *[2];
  argv[0] = program_name;
  argv[1] = NULL;
  gflags::ParseCommandLineFlags(&argc, reinterpret_cast<char ***>(&argv), true);
  EXPECT_EQ("gflags_test", std::string(gflags::ProgramInvocationShortName()));
  EXPECT_EQ("Hello world!", FLAGS_message);
  gflags::ShutDownCommandLineFlags();
}

}  // namespace gflags_test
}  // namespace third_party
