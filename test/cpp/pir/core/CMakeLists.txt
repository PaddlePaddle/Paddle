paddle_test(type_test SRCS type_test.cc DEPS common)
paddle_test(ir_attribute_test SRCS ir_attribute_test.cc)
paddle_test(ir_value_test SRCS ir_value_test.cc)
paddle_test(ir_op_test SRCS ir_op_test.cc DEPS test_dialect)
paddle_test(ir_region_test SRCS ir_region_test.cc)
paddle_test(ir_builder_test SRCS ir_builder_test.cc)
paddle_test(ir_program_test SRCS ir_program_test.cc)
paddle_test(ir_infershape_test SRCS ir_infershape_test.cc)
paddle_test(scalar_attribute_test SRCS scalar_attribute_test.cc)
paddle_test(paddle_fatal_test SRCS paddle_fatal_test.cc)

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

paddle_test(program_translator_test SRCS program_translator_test.cc)

paddle_test(add_dialect_parser_test SRCS add_dialect_parser_test.cc DEPS gtest)

paddle_test(ir_op_info_test SRCS op_info_test.cc)
paddle_test(ir_op_yaml_info_parser_test SRCS op_yaml_info_parser_test.cc)

paddle_test(ir_type_converter_test SRCS ir_type_converter_test.cc)

paddle_test(type_interface_test SRCS type_interface_test.cc DEPS test_dialect
            gtest)

paddle_test(block_operand_test SRCS block_operand_test.cc DEPS test_dialect
            gtest)

paddle_test(block_argument_test SRCS block_argument_test.cc DEPS test_dialect)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(ir_op_test)
endif()
