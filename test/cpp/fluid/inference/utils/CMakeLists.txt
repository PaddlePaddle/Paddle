paddle_test(
  infer_io_utils_tester
  SRCS
  io_utils_tester.cc
  DEPS
  infer_io_utils
  fleet_executor
  parallel_executor)
