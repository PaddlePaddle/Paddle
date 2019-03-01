cc_library(anakin_op_converter SRCS fc.cc pool2d.cc concat.cc split.cc registrar.cc DEPS anakin_engine framework_proto scope op_registry)
cc_test(test_anakin_fc SRCS test_fc_op.cc DEPS  ${FLUID_CORE_MODULES} ${GLOB_OPERATOR_DEPS}  anakin_op_converter mul_op)
cc_test(test_anakin_pool2d SRCS test_pool2d_op.cc DEPS  ${FLUID_CORE_MODULES}
   ${GLOB_OPERATOR_DEPS} anakin_op_converter pool_op pooling)
cc_test(test_anakin_concat SRCS test_concat_op.cc DEPS  ${FLUID_CORE_MODULES}
   ${GLOB_OPERATOR_DEPS} anakin_op_converter concat_op concat_and_split)

cc_test(test_anakin_split SRCS test_split_op.cc DEPS  ${FLUID_CORE_MODULES}
   ${GLOB_OPERATOR_DEPS} anakin_op_converter split_op concat_and_split)
