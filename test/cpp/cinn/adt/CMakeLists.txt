cinn_cc_test(equation_value_match_trait_test SRCS
             equation_value_match_trait_test.cc DEPS gtest glog)

cinn_cc_test(tree_test SRCS tree_test.cc DEPS gtest glog)

cinn_cc_test(
  inline_translator_test
  SRCS
  inline_translator_test.cc
  DEPS
  gtest
  glog
  absl)
