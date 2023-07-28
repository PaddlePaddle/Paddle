cc_test_old(type_test SRCS type_test.cc DEPS ir gtest)
cc_test_old(ir_attribute_test SRCS ir_attribute_test.cc DEPS ir gtest)
cc_test_old(ir_value_test SRCS ir_value_test.cc DEPS ir gtest)
cc_test_old(ir_op_test SRCS ir_op_test.cc DEPS ir gtest)
cc_test_old(ir_region_test SRCS ir_region_test.cc DEPS ir gtest)
cc_test_old(ir_builder_test SRCS ir_builder_test.cc DEPS ir gtest)
cc_test_old(
  ir_program_test
  SRCS
  ir_program_test.cc
  DEPS
  pd_dialect
  ir
  phi
  gtest)

cc_test_old(
  ir_infershape_test
  SRCS
  ir_infershape_test.cc
  DEPS
  pd_dialect
  ir
  phi
  gtest)

cc_test_old(
  ir_exe_test
  SRCS
  ir_exe_test.cc
  DEPS
  pd_op_to_kernel_pass
  program_translator
  pd_dialect
  phi_kernel_adaptor
  ir
  phi
  gtest)

cc_test_old(
  scalar_attribute_test
  SRCS
  scalar_attribute_test.cc
  DEPS
  pd_dialect
  ir
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
  pd_dialect
  ir)

cc_test_old(ir_op_info_test SRCS op_info_test.cc DEPS gtest ir)
cc_test_old(
  ir_op_yaml_info_parser_test
  SRCS
  op_yaml_info_parser_test.cc
  DEPS
  gtest
  pd_dialect
  pd_interface
  ir)

cc_test_old(
  ir_type_converter_test
  SRCS
  ir_type_converter_test.cc
  DEPS
  gtest
  program_translator
  pd_dialect
  ir)
