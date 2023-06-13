cc_test_old(type_test SRCS type_test.cc DEPS ir_shared gtest)
cc_test_old(ir_attribute_test SRCS ir_attribute_test.cc DEPS ir_shared gtest)
cc_test_old(ir_value_test SRCS ir_value_test.cc DEPS ir_shared gtest)
cc_test_old(ir_op_test SRCS ir_op_test.cc DEPS ir_shared gtest)
cc_test_old(
  ir_program_test
  SRCS
  ir_program_test.cc
  DEPS
  ir_shared
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_phi_kernel_op_test
  SRCS
  ir_phi_kernel_op_test.cc
  DEPS
  ir_shared
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_infershape_test
  SRCS
  ir_infershape_test.cc
  DEPS
  ir_shared
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_exe_test
  SRCS
  ir_exe_test.cc
  DEPS
  ir_shared
  pd_dialect
  phi
  gtest)

cc_test_old(
  scalar_attribute_test
  SRCS
  scalar_attribute_test.cc
  DEPS
  ir_shared
  pd_dialect
  gtest)

file(
  DOWNLOAD
  https://paddle-ci.gz.bcebos.com/ir_translator_test/restnet50_main.prog
  ${CMAKE_CURRENT_BINARY_DIR}/restnet50_main.prog
  EXPECTED_MD5 b64c0ad3c96d99fc37d12094623ce1ad)

cc_test_old(
  program_translator_test
  SRCS
  program_translator_test.cc
  DEPS
  program_translator
  gtest
  ir_shared
  pd_dialect)
