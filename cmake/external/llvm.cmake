include(ExternalProject)

set(LLVM_PROJECT "extern_llvm")
set(LLVM_VER     "7.0.0")

set(LLVM_TAR "llvm-${LLVM_VER}.src.tar.xz")
set(LLVM_URL "http://releases.llvm.org/${LLVM_VER}/${LLVM_TAR}")

set(LLVM_SOURCES_DIR  ${THIRD_PARTY_PATH}/llvm)
set(LLVM_DOWNLOAD_DIR ${LLVM_SOURCES_DIR}/src/${LLVM_PROJECT})

set(LLVM_BUILD_DIR  ${LLVM_DOWNLOAD_DIR}/llvm-${LLVM_VER}.src/build)
# LLVM takes a long time to compile
set(LLVM_BUILD_CMD  mkdir -p ${LLVM_BUILD_DIR} && cd ${LLVM_BUILD_DIR} && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j16) 

ExternalProject_Add(
  ${LLVM_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  DOWNLOAD_DIR          ${LLVM_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND      wget --no-check-certificate  ${LLVM_URL} -c -q -O ${LLVM_TAR} && tar xf ${LLVM_TAR}
  DOWNLOAD_NO_PROGRESS  1
  PREFIX                ${LLVM_SOURCES_DIR}
  CONFIGURE_COMMAND     ""
  BUILD_COMMAND         ${LLVM_BUILD_CMD}
  INSTALL_COMMAND       ""
  UPDATE_COMMAND        ""
)
