#include <gflags/gflags.h>
#include <iostream>
#include <sstream>

DEFINE_bool(color, true, "Whether to turn on pretty log");

namespace paddle {
namespace framework {
namespace analysis {

namespace logging {

inline const char* black() { return FLAGS_color ? "\e[30m" : ""; }
inline const char* red() { return FLAGS_color ? "\e[31m" : ""; }
inline const char* b_red() { return FLAGS_color ? "\e[41m" : ""; }
inline const char* green() { return FLAGS_color ? "\e[32m" : ""; }
inline const char* yellow() { return FLAGS_color ? "\e[33m" : ""; }
inline const char* blue() { return FLAGS_color ? "\e[34m" : ""; }
inline const char* purple() { return FLAGS_color ? "\e[35m" : ""; }
inline const char* cyan() { return FLAGS_color ? "\e[36m" : ""; }
inline const char* light_gray() { return FLAGS_color ? "\e[37m" : ""; }
inline const char* white() { return FLAGS_color ? "\e[37m" : ""; }
inline const char* light_red() { return FLAGS_color ? "\e[91m" : ""; }
inline const char* dim() { return FLAGS_color ? "\e[2m" : ""; }
inline const char* bold() { return FLAGS_color ? "\e[1m" : ""; }
inline const char* underline() { return FLAGS_color ? "\e[4m" : ""; }
inline const char* blink() { return FLAGS_color ? "\e[5m" : ""; }
inline const char* reset() { return FLAGS_color ? "\e[0m" : ""; }

struct LogStream {
  enum Level { INFO, WARNING, SUCCESS };
  LogStream(const char* header) : os_(std::cerr) { os_ << header << " "; }

  ~LogStream() { os_ << reset() << "\n"; }

  std::ostream& stream() { return os_; }

  void operator=(const LogStream&) = delete;
  LogStream(const LogStream&) = delete;

 private:
  std::ostream& os_;
};

}  // namespace logging

}  // namespace analysis
}  // namespace framework
}  // namespace paddle

#define PLOG(LEVEL) PLOG_##LEVEL
#define PLOG_INFO                                    \
  ::paddle::framework::analysis::logging::LogStream( \
      paddle::framework::analysis::logging::black()) \
      .stream()
#define PLOG_SUCCESS                                 \
  ::paddle::framework::analysis::logging::LogStream( \
      paddle::framework::analysis::logging::green()) \
      .stream()

#define PLOG_WARNING                                 \
  ::paddle::framework::analysis::logging::LogStream( \
      paddle::framework::analysis::logging::bold())  \
      .stream()

#define PLOG_ERROR                                   \
  ::paddle::framework::analysis::logging::LogStream( \
      paddle::framework::analysis::logging::red())   \
      .stream()
