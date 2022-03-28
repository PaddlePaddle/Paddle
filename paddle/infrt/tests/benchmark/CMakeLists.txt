cc_test_tiny(benchmark_exe SRCS benchmark.cc DEPS infrt ${MLIR_IR_LIBS})
cc_test_tiny(resnet50_infrt SRCS resnet50_infrt.cc DEPS infrt ${MLIR_IR_LIBS}) 

set(inference_deps ${analysis_deps} paddle_inference_api analysis naive_executor ${GLOB_PASS_LIB})
cc_test_tiny(fluid_benchmark SRCS fluid_benchmark.cc DEPS ${inference_deps} 
zero_copy_tensor ir_pass_manager op_compatible_info infer_io_utils analysis_predictor init)

cc_test_tiny(resnet50_benchmark SRCS resnet50_benchmark.cc DEPS ${inference_deps} 
zero_copy_tensor ir_pass_manager op_compatible_info infer_io_utils analysis_predictor init)

#add_executable(benchmark_exe benchmark.cc)
#target_link_libraries(benchmark_exe infrt ${MLIR_IR_LIBS})
