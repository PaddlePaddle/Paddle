add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sample_trainer_config.conf
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(copy_trainer_conf ALL DEPENDS sample_trainer_config.conf)

set(PYTHON_PATH 
   ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d 
   ${PADDLE_BINARY_DIR}/python/:${PADDLE_BINARY_DIR}/paddle/trainer/tests)
function(trainer_test TARGET)
  add_unittest_without_exec(${TARGET} ${TARGET}.cpp)
  add_test(NAME ${TARGET}
    COMMAND ${PYTHON_PATH} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}
      WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle/)
endfunction()

trainer_test(test_Compare)
trainer_test(test_PyDataProviderWrapper)
trainer_test(test_recurrent_machine_generation)
trainer_test(test_Trainer)

############### test_TrainerOnePass ##########################
if(WITH_PYTHON)
  # only run test_TrainerOnePass when PYTHON is enabled, because train one pass
  # is using PyDataProvider2.
  add_unittest_without_exec(test_TrainerOnePass
      test_TrainerOnePass.cpp)
  add_test(NAME test_TrainerOnePass
    COMMAND ${PYTHON_PATH} ${PADDLE_SOURCE_DIR}/paddle/.set_port.sh -p port 
          ${CMAKE_CURRENT_BINARY_DIR}/test_TrainerOnePass
      WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle/)
endif()

#################### test_config_parser #########################
add_test(NAME test_config_parser
  COMMAND ${PYTHON_PATH} ${PYTHON_EXECUTABLE} 
        ${PADDLE_SOURCE_DIR}/paddle/trainer/tests/config_parser_test.py
    WORKING_DIRECTORY ${PADDLE_BINARY_DIR}/paddle/)
