set(AP_COMMON_DEPS "")
if(WITH_ONEDNN)
  list(APPEND AP_COMMON_DEPS onednn)
endif()

file(GLOB_RECURSE axpr_srcs "src/axpr/*.cc")
set(axpr_deps common)
cc_library(
  axpr
  SRCS ${axpr_srcs}
  DEPS ${AP_COMMON_DEPS} ${axpr_deps})

file(GLOB_RECURSE ap_drr_srcs "src/drr/*.cc")
set(ap_drr_deps axpr)
cc_library(
  ap_drr
  SRCS ${ap_drr_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_drr_deps})

file(GLOB_RECURSE ap_code_module_srcs "src/code_module/*.cc")
set(ap_code_module_deps axpr)
cc_library(
  ap_code_module
  SRCS ${ap_code_module_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_code_module_deps})

file(GLOB_RECURSE ap_code_gen_srcs "src/code_gen/*.cc")
set(ap_code_gen_deps axpr ap_code_module)
cc_library(
  ap_code_gen
  SRCS ${ap_code_gen_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_code_gen_deps})

file(GLOB_RECURSE ap_kernel_dispatch_srcs "src/kernel_dispatch/*.cc")
set(ap_kernel_dispatch_deps axpr ap_code_module ap_code_gen)
cc_library(
  ap_kernel_dispatch
  SRCS ${ap_kernel_dispatch_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_kernel_dispatch_deps})

file(GLOB_RECURSE ap_phi_srcs "src/paddle/phi/*.cc")
set(ap_phi_deps axpr ap_code_module ap_code_gen ap_kernel_dispatch)
cc_library(
  ap_phi
  SRCS ${ap_phi_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_phi_deps})

file(GLOB_RECURSE ap_pir_srcs "src/paddle/pir/*.cc")
set(ap_pir_deps axpr ap_drr)
cc_library(
  ap_pir
  SRCS ${ap_pir_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_pir_deps})

file(GLOB_RECURSE ap_reified_drr_srcs "src/reified_drr/*.cc")
set(ap_reified_drr_deps axpr ap_drr ap_code_module ap_code_gen)
cc_library(
  ap_reified_drr
  SRCS ${ap_reified_drr_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_reified_drr_deps})

file(GLOB_RECURSE ap_pass_srcs "src/paddle/pass/*.cc")
set(ap_pass_deps axpr ap_pir ap_drr ap_code_module ap_code_gen ap_reified_drr)
cc_library(
  ap_pass
  SRCS ${ap_pass_srcs}
  DEPS ${AP_COMMON_DEPS} ${ap_pass_deps})
