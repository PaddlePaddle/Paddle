function(paddle_resolve_llvm_path out_var)
  set(_llvm_path "${${out_var}}")
  if(EXISTS "${_llvm_path}/bin/llvm-config")
    set(${out_var}
        "${_llvm_path}"
        PARENT_SCOPE)
    return()
  endif()

  file(
    GLOB _llvm_path_candidates
    LIST_DIRECTORIES true
    "${_llvm_path}/clang+llvm-*")
  foreach(_llvm_path_candidate IN LISTS _llvm_path_candidates)
    if(EXISTS "${_llvm_path_candidate}/bin/llvm-config")
      set(${out_var}
          "${_llvm_path_candidate}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR "Cannot find bin/llvm-config under ${_llvm_path}")
endfunction()

function(paddle_fix_llvm_support_target)
  if(NOT TARGET LLVMSupport)
    return()
  endif()

  get_target_property(_llvm_support_libs LLVMSupport INTERFACE_LINK_LIBRARIES)
  if(NOT _llvm_support_libs OR _llvm_support_libs STREQUAL
                               "_llvm_support_libs-NOTFOUND")
    return()
  endif()

  set(_updated_llvm_support_libs)
  set(_replaced_tinfo_path OFF)
  set(_replaced_zlib_target OFF)
  foreach(_llvm_support_lib IN LISTS _llvm_support_libs)
    if(_llvm_support_lib STREQUAL "ZLIB::ZLIB" AND TARGET zlib)
      list(APPEND _updated_llvm_support_libs zlib)
      set(_replaced_zlib_target ON)
    elseif(_llvm_support_lib MATCHES "^/.*/libtinfo\\.so$")
      find_library(PADDLE_LLVM_TINFO_LIBRARY NAMES tinfo ncursesw ncurses)
      if(PADDLE_LLVM_TINFO_LIBRARY)
        list(APPEND _updated_llvm_support_libs "${PADDLE_LLVM_TINFO_LIBRARY}")
        set(_replaced_tinfo_path ON)
      elseif(EXISTS "${_llvm_support_lib}")
        list(APPEND _updated_llvm_support_libs "${_llvm_support_lib}")
      else()
        message(
          FATAL_ERROR
            "LLVM package references ${_llvm_support_lib}, but no compatible tinfo/ncurses library was found."
        )
      endif()
    else()
      list(APPEND _updated_llvm_support_libs "${_llvm_support_lib}")
    endif()
  endforeach()

  if(_replaced_tinfo_path OR _replaced_zlib_target)
    set_target_properties(
      LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES
                             "${_updated_llvm_support_libs}")
    if(_replaced_tinfo_path)
      message(STATUS "Replaced LLVM LLVMSupport tinfo path with "
                     "${PADDLE_LLVM_TINFO_LIBRARY}")
    endif()
    if(_replaced_zlib_target)
      message(STATUS "Replaced LLVM LLVMSupport ZLIB::ZLIB dependency with "
                     "Paddle zlib target")
    endif()
  endif()
endfunction()

function(paddle_select_mlir_standard_lib out_var)
  if(TARGET MLIRStandard)
    set(${out_var}
        MLIRStandard
        PARENT_SCOPE)
  elseif(TARGET MLIRStandardOps)
    set(${out_var}
        MLIRStandardOps
        PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Cannot find MLIR standard dialect library target.")
  endif()
endfunction()
