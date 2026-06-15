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
