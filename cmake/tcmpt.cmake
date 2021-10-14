# TODO(chenweihang): keep message comment for debuging, remove it if needless
function(kernel_instantiate TARGET)
    set(target_file ${CURRENT_BINARY_DIR}/${TARGET}.tmp CACHE INTERNAL "${CURRENT_BINARY_DIR}/${TARGET} file")
    set(target_file_final ${CURRENT_BINARY_DIR}/${TARGET})
    file(READ ${TARGET} TARGET_CONTENT)
    file(WRITE ${target_file} ${TARGET_CONTENT})
    string(REGEX MATCHALL "void [A-Z][A-Za-z0-9_]+\\(.[^\\)]+\\)" func_signatures ${TARGET_CONTENT})
    # message(STATUS "FUNCS: ${func_signatures}")
    string(REGEX MATCHALL "PT_REGISTER_KERNEL\\(.[^\\)]+\\) \\{" func_registrars ${TARGET_CONTENT})
    # message(STATUS "REGISTRARS: ${func_registrars}")
    set(instantiate_context "")
    foreach(signature ${func_signatures})
        # message(STATUS "FUNC: ${signature}")
        list(POP_FRONT func_registrars registrar)
        # message(STATUS "REG: ${registrar}")
        string(REGEX MATCHALL "[a-z0-9_:]+(,|\\))" dtypes ${registrar})
        # message(STATUS "DTYPES: ${dtypes}")
        list(REMOVE_AT dtypes 0)
        # message(STATUS "REMOVED DTYPES: ${dtypes}")
        foreach(dtype ${dtypes})
            string(REGEX REPLACE ",|\\)" "" dtype ${dtype})
            # message(STATUS "DTYPE: ${dtype}")
            string(REGEX MATCH "[A-Z][A-Za-z0-9]+\\(" func_name ${signature})
            string(REPLACE "(" "" func_name ${func_name})
            # message(STATUS "FUNC NAME: ${func_name}")
            string(REGEX REPLACE "${func_name}" "pt::${func_name}<${dtype}>" inst_signature ${signature})
            # append namespace
            string(REPLACE "CPUContext" "pt::CPUContext" inst_signature ${inst_signature})
            string(REPLACE "CUDAContext" "pt::CUDAContext" inst_signature ${inst_signature})
            string(REPLACE "DenseTensor" "pt::DenseTensor" inst_signature ${inst_signature})
            # TODO(chenweihang): adapt SelectedRows after adding it
            # string(REPLACE "SelectedRowsTensor" "pt::SelectedRowsTensor" inst_signature ${inst_signature})
            # message(STATUS "INST FUNC: ${inst_signature}")
            string(APPEND instantiate_context "template ${inst_signature};\n")
        endforeach()
    endforeach()
    # message(STATUS "INST CONTENT: ${instantiate_context}")
    file(APPEND ${target_file} "${instantiate_context}\n")
    # copy_if_different(${target_file} ${target_file_final})
    string(REPLACE "." "_" cmd_name ${TARGET})
    # this is a dummy target for custom command, should always be run firstly to update ${target_file_final}
    # TODO(chenweihang): nameing rule need to enchance
    add_custom_target(copy_${cmd_name}_command ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${target_file} ${target_file_final}
        COMMENT "copy_if_different ${target_file_final}"
        VERBATIM
    )
    add_dependencies(extern_glog copy_${cmd_name}_command)
endfunction()