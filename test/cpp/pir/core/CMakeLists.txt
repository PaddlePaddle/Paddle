paddle_test(type_test SRCS type_test.cc DEPS pir op_dialect_vjp)
paddle_test(ir_attribute_test SRCS ir_attribute_test.cc DEPS pir)
paddle_test(ir_value_test SRCS ir_value_test.cc DEPS pir)
paddle_test(
  ir_op_test
  SRCS
  ir_op_test.cc
  DEPS
  pir
  test_dialect
  op_dialect_vjp)
paddle_test(ir_region_test SRCS ir_region_test.cc DEPS pir)
paddle_test(ir_builder_test SRCS ir_builder_test.cc DEPS pir)
cc_test_old(
  ir_program_test
  SRCS
  ir_program_test.cc
  DEPS
  op_dialect_vjp
  pir
  phi
  common
  gtest)

cc_test_old(
  ir_infershape_test
  SRCS
  ir_infershape_test.cc
  DEPS
  op_dialect_vjp
  pir
  phi
  common
  gtest)

cc_test_old(
  scalar_attribute_test
  SRCS
  scalar_attribute_test.cc
  DEPS
  op_dialect_vjp
  pir
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

file(
  DOWNLOAD
  https://paddle-ci.gz.bcebos.com/ir_translator_test/conditional_block_test.prog
  ${CMAKE_CURRENT_BINARY_DIR}/conditional_block_test.prog
  EXPECTED_MD5 cf9dc869ca7f69e2d57b38dbf8427134)

file(
  DOWNLOAD https://paddle-ci.gz.bcebos.com/ir_translator_test/while_op_test.prog
  ${CMAKE_CURRENT_BINARY_DIR}/while_op_test.prog
  EXPECTED_MD5 290164ae52a496332b0be5829fc93bcd)

copy_if_different(${CMAKE_CURRENT_SOURCE_DIR}/TestParserText.txt
                  ${CMAKE_CURRENT_BINARY_DIR}/TestParserText.txt)

cc_test_old(
  program_translator_test
  SRCS
  program_translator_test.cc
  DEPS
  program_translator
  gtest
  op_dialect_vjp
  pir)

cc_test_old(
  add_dialect_parser_test
  SRCS
  add_dialect_parser_test.cc
  DEPS
  gtest
  op_dialect_vjp
  pir)

cc_test(
  ir_parser_test
  SRCS ir_parser_test.cc
  DEPS gtest op_dialect_vjp pir)

cc_test_old(ir_op_info_test SRCS op_info_test.cc DEPS gtest pir)
cc_test_old(
  ir_op_yaml_info_parser_test
  SRCS
  op_yaml_info_parser_test.cc
  DEPS
  gtest
  op_dialect
  op_dialect_vjp
  pir)

cc_test_old(
  ir_type_converter_test
  SRCS
  ir_type_converter_test.cc
  DEPS
  gtest
  program_translator
  op_dialect_vjp
  pir)

cc_test_old(
  type_interface_test
  SRCS
  type_interface_test.cc
  DEPS
  pir
  test_dialect
  gtest
  op_dialect_vjp)

cc_test_old(
  block_operand_test
  SRCS
  block_operand_test.cc
  DEPS
  test_dialect
  gtest
  pir)

paddle_test(block_argument_test SRCS block_argument_test.cc DEPS test_dialect
            pir)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(ir_op_test)
endif()
