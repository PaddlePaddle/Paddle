set(tensorrt_engine_op_deps tensorrt_engine tensorrt_converter infer_io_utils
                            analysis_helper)

op_library(tensorrt_engine_op DEPS ${tensorrt_engine_op_deps})
