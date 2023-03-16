
# Compile

## 1. compile llvm

llvm commit id: 10939d1d580b9d3c9c2f3539c6bdb39f408179c0
```
cd llvm-project
mkdir build && cd build
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On
ninja
```

## 2. compile googletest

```
git clone https://github.com/google/googletest.git
cd googletest
git checkout v1.13.0
mkdir build && mkdir install
cmake -B build/ -DCMAKE_INSTALL_PREFIX=install/ .
cmake --build build/ -j 16
cmake --install build/
```

## 2. compile infra

```bash
mkdir build && cd build
cmake .. -GNinja -DLLVM_PATH=...llvm-project/build -DGTEST_PATH=...googletest/install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja
```
