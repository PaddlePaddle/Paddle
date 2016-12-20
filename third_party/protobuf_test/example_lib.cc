#include "third_party/protobuf_test/example_lib.h"

namespace third_party {
namespace protobuf_test {

std::string get_greet(const Greeting& who) { return "Hello " + who.name(); }

}  // namespace protobuf_test
}  // namespace thrid_party
