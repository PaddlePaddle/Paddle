cc_library(benchmark SRCS benchmark.cc DEPS enforce)
cc_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)
cc_binary(visualizer SRCS visualizer.cc DEPS analysis
    paddle_pass_builder ir_pass_manager pass graph_viz_pass analysis_passes)
