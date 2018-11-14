unset(analysis_deps CACHE)
set(analysis_deps # analysis_deps can be extended accross the project
        framework_proto proto_desc graph pass paddle_fluid_api executor pretty_log
        ir_pass_manager
        CACHE INTERNAL "")

add_subdirectory(ir_passes)
add_subdirectory(passes)

cc_library(ir_pass_manager SRCS ir_pass_manager.cc DEPS graph pass ${INFER_IR_PASSES})

cc_library(argument SRCS argument.cc DEPS scope proto_desc)
cc_library(analysis_pass SRCS analysis_pass.cc DEPS proto_desc)

cc_library(analysis SRCS
  analyzer.cc
  helper.cc
  analysis_pass
  DEPS ${analysis_deps}
  )

cc_test(test_dot SRCS dot_tester.cc DEPS analysis)

if(WIN32)
    target_link_libraries(inference_analyzer shlwapi)
endif(WIN32)

function (inference_analysis_test TARGET)
    if(WITH_TESTING)
        set(options "")
        set(oneValueArgs "")
        set(multiValueArgs SRCS ARGS EXTRA_DEPS)
        cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        set(mem_opt "")
        if(WITH_GPU)
            set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
        endif()
        cc_test(${TARGET}
                SRCS "${analysis_test_SRCS}"
                DEPS analysis pass ${GLOB_PASS_LIB} ${analysis_test_EXTRA_DEPS}
                ARGS --inference_model_dir=${PYTHON_TESTS_DIR}/book/word2vec.inference.model ${mem_opt} ${analysis_test_ARGS})
        set_tests_properties(${TARGET} PROPERTIES DEPENDS test_word2vec)
    endif(WITH_TESTING)
endfunction(inference_analysis_test)

inference_analysis_test(test_analyzer SRCS analyzer_tester.cc EXTRA_DEPS paddle_inference_api)
