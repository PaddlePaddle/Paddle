#include <gflags/gflags.h>
#include <iostream>
#include <sstream>

DEFINE_bool(color, ture, "Whether to turn on pretty log");

namespace paddle {
namespace framework {
namespace analysis {

namespace logging {
inline const char* black() { return FLAG_color ? "\e[30m" : ""; }
inline const char* red() { return FLAG_color ? "\e[31m" : ""; }
inline const char* b_red() { return FLAG_color ? "\e[41m" : ""; }
inline const char* green() { return FLAG_color ? "\e[32m" : ""; }
inline const char* yellow() { return FLAG_color ? "\e[33m" : ""; }
inline const char* blue() { return FLAG_color ? "\e[34m" : ""; }
inline const char* purple() { return FLAG_color ? "\e[35m" : ""; }
inline const char* cyan() { return FLAG_color ? "\e[36m" : ""; }
inline const char* light_gray() { return FLAG_color ? "\e[37m" : ""; }
inline const char* white() { return FLAG_color ? "\e[37m" : ""; }
inline const char* light_red() { return FLAG_color ? "\e[91m" : ""; }
inline const char* dim() { return FLAG_color ? "\e[2m" : ""; }
inline const char* bold() { return FLAG_color ? "\e[1m" : ""; }
inline const char* underline() { return FLAG_color ? "\e[4m" : ""; }
inline const char* blink() { return FLAG_color ? "\e[5m" : ""; }
inline const char* reset() { return FLAG_color ? "\e[0m" : ""; }

struct LogStream {
  enum Level { INFO, WARNING, SUCCESS };
  LogStream() : os_(std::cerr) {}

  ~LogStream() { os_ << reset() << "\n"; }

  std::ostream& stream() { return os_; }

  void operator=(const LogStream&) = delete;
  LogStream(const LogStream&) = delete;

 private:
  std::ostream& os_;
};
}

}  // namespace analysis
}  // namespace framework
}  // namespace paddle

#define PLOG(LEVEL) PLOG##LEVEL
#define PLOG_INFO ::paddle::framework::analysis::logging::LogStream().stream()
#define PLOG_SUCCESS \
  ::paddle::framework::analysis::logging::LogStream().stream() << green()
#define PLOG_WARNING                                                       \
  ::paddle::framework::analysis::logging::LogStream().stream()             \
      << bold() #define PLOG_ERROR::paddle::framework::analysis::logging:: \
             LogStream()                                                   \
                 .stream()                                                 \
      << red();
