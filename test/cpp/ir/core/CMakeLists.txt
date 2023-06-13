cc_test_old(type_test SRCS type_test.cc DEPS ir gtest)
cc_test_old(ir_attribute_test SRCS ir_attribute_test.cc DEPS ir gtest)
cc_test_old(ir_value_test SRCS ir_value_test.cc DEPS ir gtest)
cc_test_old(ir_op_test SRCS ir_op_test.cc DEPS ir gtest)
cc_test_old(
  ir_program_test
  SRCS
  ir_program_test.cc
  DEPS
  ir
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_phi_kernel_op_test
  SRCS
  ir_phi_kernel_op_test.cc
  DEPS
  ir
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_infershape_test
  SRCS
  ir_infershape_test.cc
  DEPS
  ir
  pd_dialect
  phi
  gtest)

cc_test_old(
  ir_exe_test
  SRCS
  ir_exe_test.cc
  DEPS
  ir
  pd_dialect
  phi
  gtest)

cc_test_old(
  scalar_attribute_test
  SRCS
  scalar_attribute_test.cc
  DEPS
  ir
  pd_dialect
  gtest)

file(
  DOWNLOAD https://paddle-ci.gz.bcebos.com/ir_translator_test/resnet50_main.prog
  ${CMAKE_CURRENT_BINARY_DIR}/resnet50_main.prog
  EXPECTED_MD5 b64c0ad3c96d99fc37d12094623ce1ad)

file(
  DOWNLOAD
  https://paddle-ci.gz.bcebos.com/ir_translator_test/resnet50_startup.prog
  ${CMAKE_CURRENT_BINARY_DIR}/resnet50_startup.prog
  EXPECTED_MD5 6affc5f40f0f0bb84d956919b95eaf50)

cc_test_old(
  program_translator_test
  SRCS
  program_translator_test.cc
  DEPS
  program_translator
  gtest
  ir
  pd_dialect)

cc_test_old(ir_op_info_test SRCS op_info_test.cc DEPS gtest ir)
