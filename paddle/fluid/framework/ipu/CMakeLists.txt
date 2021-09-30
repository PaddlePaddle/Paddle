set(POPART_CANONICALIZATION_SRC
    "popart_canonicalization/canonicalization_utils.cc"
    "popart_canonicalization/op_builder.cc"
    "popart_canonicalization/post_canonicalization.cc"
    "popart_canonicalization/activation_ops.cc"
    "popart_canonicalization/elementwise_ops.cc"
    "popart_canonicalization/logic_ops.cc"
    "popart_canonicalization/math_ops.cc"
    "popart_canonicalization/nn_ops.cc"
    "popart_canonicalization/tensor_ops.cc"
    "popart_canonicalization/other_ops.cc"
    "popart_canonicalization/reduce_ops.cc"
    "popart_canonicalization/search_ops.cc"
)

cc_library(ipu_device SRCS device.cc DEPS enforce popart)
cc_library(ipu_utils SRCS ipu_utils.cc DEPS memory framework_proto popart)
cc_library(ipu_strategy SRCS ipu_strategy.cc DEPS popart graph framework_proto enforce)
cc_library(ipu_optimizer SRCS ipu_optimizer.cc DEPS popart enforce)
cc_library(ipu_executor SRCS ipu_executor.cc DEPS ipu_optimizer ipu_utils popart graph framework_proto)
cc_library(popart_canonicalization_utils SRCS ${POPART_CANONICALIZATION_SRC} DEPS framework_proto enforce ipu_utils)
cc_library(ipu_compiler SRCS ipu_compiler.cc DEPS popart graph ipu_utils graph_helper)
cc_library(ipu_backend SRCS ipu_backend.cc DEPS popart ipu_compiler graph framework_proto enforce ipu_utils ipu_strategy ipu_device ipu_executor graph_helper)
