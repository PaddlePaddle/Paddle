file(GLOB_RECURSE SERIALIZE_DESERIALIZE_CPP_SOURCES "*.cc")

include_directories(pir_save_load PRIVATE
                    ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json/include/)
include_directories(pir_save_load PRIVATE
                    ${PADDLE_BINARY_DIR}/third_party/install/yaml-cpp/include/)

link_directories(${PADDLE_BINARY_DIR}/third_party/install/yaml-cpp/lib/)
target_compile_definitions(yaml INTERFACE YAML_CPP_STATIC_DEFINE)
if(LINUX)
  link_libraries(stdc++fs)
endif()

file(GLOB_RECURSE YAML_PATCH_FILES "*.yaml")
# change pir version when new patches are added
add_definitions(-DDEVELOP_VERSION=1)
add_definitions(-DRELEASE_VERSION=1)
set(TEMPLATE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/patch/template.h.in)
set(PATCH_HEADER ${CMAKE_CURRENT_BINARY_DIR}/patch/patch.h)

configure_file(${TEMPLATE_FILE} ${PATCH_HEADER} @ONLY)
file(WRITE "${PATCH_HEADER}"
     "#include <map>\n#include <string>\n\n"
     "const std::map<std::string, std::string> yaml_files = {\n")

foreach(PATCH_FILE ${YAML_PATCH_FILES})
  get_filename_component(FILENAME "${PATCH_FILE}" NAME_WE)
  file(READ ${PATCH_FILE} FILE_CONTENT)
  set(CONTENT "R\"(${FILE_CONTENT})\"")
  file(APPEND "${PATCH_HEADER}" "{ \"${FILENAME}\", ${CONTENT} },\n")
endforeach()

file(APPEND "${PATCH_HEADER}" "};\n")

cc_library(
  pir_save_load
  SRCS ${SERIALIZE_DESERIALIZE_CPP_SOURCES} ${PATCH_HEADER}
  DEPS op_dialect phi json yaml)
