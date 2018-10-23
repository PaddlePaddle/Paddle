cc_library(ir_graph_build_pass SRCS ir_graph_build_pass.cc DEPS analysis_pass argument ir_pass_manager)
cc_library(ir_analysis_pass SRCS ir_analysis_pass.cc DEPS analysis_pass argument ir_pass_manager)
cc_library(analysis_passes SRCS passes.cc DEPS ir_graph_build_pass ir_analysis_pass)

set(analysis_deps ${analysis_deps}
        ir_graph_build_pass
        ir_analysis_pass
        analysis_passes
        CACHE INTERNAL "")
