message(STATUS "===== hello!! ")

add_executable(benchmark_exe benchmark.cc)
target_link_libraries(benchmark_exe infrt ${MLIR_IR_LIBS})
