if(WITH_NEWIR)
  cc_test_old(type_test SRCS type_test.cc DEPS new_ir gtest)
  cc_test_old(ir_attribute_test SRCS ir_attribute_test.cc DEPS new_ir gtest)
  cc_test_old(ir_value_test SRCS ir_value_test.cc DEPS new_ir gtest)
  cc_test_old(ir_op_test SRCS ir_op_test.cc DEPS new_ir gtest)
  cc_test_old(ir_program_test SRCS ir_program_test.cc DEPS new_ir gtest)

  add_custom_target(
    download_rest50_program_for_program_translator
    COMMAND
      wget -nc --no-check-certificate
      https://paddle-ci.gz.bcebos.com/ir_translator_test/restnet50_main.prog
    WORKING_DIRECTORY "./")
  cc_test_old(
    program_translator_test
    SRCS
    program_translator_test.cc
    DEPS
    program_translator
    gtest
    new_ir
    paddle_dialect)
  add_dependencies(program_translator_test
                   download_rest50_program_for_program_translator)

endif()
