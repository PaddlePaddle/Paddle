
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

## 2. compile infra

```bash
mkdir build && cd build
cmake .. -GNinja -DLLVM_PATH=...llvm-project/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja
```
