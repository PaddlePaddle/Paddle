################# test_Compare ############################
add_unittest_without_exec(test_Compare
    test_Compare.cpp)
add_test(NAME test_Compare
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python
        ${CMAKE_CURRENT_BINARY_DIR}/test_Compare
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

################# test_Trainer ###########################
add_unittest_without_exec(test_Trainer
    test_Trainer.cpp)
add_test(NAME test_Trainer
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/paddle/trainer/tests/gen_proto_data.py &&
        ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${CMAKE_CURRENT_BINARY_DIR}/test_Trainer
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

############### test_TrainerOnePass ##########################
if(WITH_PYTHON)
  # only run test_TrainerOnePass when PYTHON is enabled, because train one pass
  # is using PyDataProvider2.
  add_unittest_without_exec(test_TrainerOnePass
      test_TrainerOnePass.cpp)
  add_test(NAME test_TrainerOnePass
    COMMAND  ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d
          ${PADDLE_SOURCE_DIR}/python/:${PADDLE_SOURCE_DIR}/paddle/trainer/tests
          ${PADDLE_SOURCE_DIR}/paddle/.set_port.sh -p port ${CMAKE_CURRENT_BINARY_DIR}/test_TrainerOnePass
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)
endif()
################ test_CompareTwoNets ######################
add_unittest_without_exec(test_CompareTwoNets
    test_CompareTwoNets.cpp)
add_test(NAME test_CompareTwoNets
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${CMAKE_CURRENT_BINARY_DIR}/test_CompareTwoNets
            --config_file_a=trainer/tests/sample_trainer_config_qb_rnn.conf --config_file_b=trainer/tests/sample_trainer_config_rnn.conf
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

############### test_CompareTwoOpts ###################
add_unittest_without_exec(test_CompareTwoOpts
    test_CompareTwoOpts.cpp)
add_test(NAME test_CompareTwoOpts
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${CMAKE_CURRENT_BINARY_DIR}/test_CompareTwoOpts
            --config_file_a=trainer/tests/sample_trainer_config_opt_a.conf --config_file_b=trainer/tests/sample_trainer_config_opt_b.conf
            --num_passes=1 --need_high_accuracy=0
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

################# test_CompareSparse ##################
add_unittest_without_exec(test_CompareSparse
    test_CompareSparse.cpp)
if(NOT ON_TRAVIS)
  add_test(NAME test_CompareSparse
    COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
              ./.set_port.sh -p port -n 6
                  ${CMAKE_CURRENT_BINARY_DIR}/test_CompareSparse
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)
endif()
################# test_recurrent_machine_generation ###############
add_unittest_without_exec(test_recurrent_machine_generation
    test_recurrent_machine_generation.cpp)
add_test(NAME test_recurrent_machine_generation
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${CMAKE_CURRENT_BINARY_DIR}/test_recurrent_machine_generation
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

#################### test_PyDataProviderWrapper #########################
add_unittest_without_exec(test_PyDataProviderWrapper
    test_PyDataProviderWrapper.cpp)

add_test(NAME test_PyDataProviderWrapper
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d
        ${PADDLE_SOURCE_DIR}/python/:${PADDLE_SOURCE_DIR}/paddle/trainer/tests
        ${CMAKE_CURRENT_BINARY_DIR}/test_PyDataProviderWrapper
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)

#################### test_config_parser #########################
add_test(NAME test_config_parser
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_SOURCE_DIR}/python/
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/paddle/trainer/tests/config_parser_test.py
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/paddle/)
