# A small utility function which generates a C-header from an input file
function(FILE_TO_C_STRING FILENAME VARIABLE_NAME OUTPUT_STRING ZERO_TERMINATED)
  FILE(READ "${FILENAME}" HEX_INPUT HEX)
  if (${ZERO_TERMINATED})
    string(APPEND HEX_INPUT "00")
  endif()

  string(REGEX REPLACE "(....)" "\\1\n" HEX_OUTPUT ${HEX_INPUT})
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," HEX_OUTPUT ${HEX_OUTPUT})

  set(HEX_OUTPUT "static char const ${VARIABLE_NAME}[] = {\n  ${HEX_OUTPUT}\n};\n")

  set(${OUTPUT_STRING} "${HEX_OUTPUT}" PARENT_SCOPE)
endfunction()

# message("Create header file for ${FILE_IN}")
# message("Create header file for ${FILE_OUT}")
file_to_c_string(${FILE_IN} ${VARIABLE_NAME} OUTPUT_STRING ZERO_TERMINATED)

set(RESULT "#pragma once\n")
string(APPEND RESULT "namespace cutlass {\n")
string(APPEND RESULT "namespace nvrtc {\n")
string(APPEND RESULT "${OUTPUT_STRING}")
string(APPEND RESULT "} // namespace nvrtc\n")
string(APPEND RESULT "} // namespace cutlass\n")
file(WRITE "${FILE_OUT}" "${RESULT}")
