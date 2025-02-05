cc_library(
  test_dialect
  SRCS test_dialect.cc test_op.cc test_trait.cc test_interface.cc
       test1_dialect.cc
  DEPS gtest)

get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
target_link_libraries(test_dialect $<TARGET_LINKER_FILE:${paddle_lib}>)
