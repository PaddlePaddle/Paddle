import lit.formats

config.name = "MLIR tests"
config.test_format = lit.formats.ShTest(True)
config.llvm_tools_dir = "/home/chunwei/project/Paddle/build/third_party/install/llvm/bin"
config.llvm_tools_dir = "/home/chunwei/project/Paddle/build/third_party/install/llvm/lib"
test_bin = "/home/chunwei/project/Paddle/build/paddle/infrt/dialect/"
llvm_bin = "/home/chunwei/project/Paddle/build/third_party/install/llvm/bin/"
config.environment['PATH'] = os.path.pathsep.join((test_bin, llvm_bin, config.environment['PATH']))

config.suffixes = ['.mlir']
