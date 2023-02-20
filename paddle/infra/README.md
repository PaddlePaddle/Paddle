
mkdir build && cd build

cmake .. -GNinja -DLLVM_PATH=...llvm-project/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
