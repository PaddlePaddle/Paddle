cc_library(auc_calculator SRCS auc_calculator.cc DEPS lodtensor_printer)
cc_library(wuauc_calculator SRCS wuauc_calculator.cc DEPS fleet_wrapper lodtensor_printer)
cc_library(pn_calculator SRCS pn_calculator.cc DEPS fleet_wrapper lodtensor_printer)
