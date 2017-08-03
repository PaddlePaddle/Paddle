#include <string>

namespace paddle {
namespace framework {
namespace op_helpers {

/*
 * Generate the gradient variable's name of a forward varialbe.
 *
 * If a variable's name has a certain suffix, it means that the
 * variable is the gradient of another varibale.
 * e.g. Variable "x@GRAD" is the gradient of varibale "x".
 */
inline std::string GenGradName(const std::string& var) {
  static const std::string suffix{"@GRAD"};
  return var + suffix;
}

}  // namespace op_helpers
}  // namespace framework
}  // namespace paddle
