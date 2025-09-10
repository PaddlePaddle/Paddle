python -c "import wget;wget.download('https://paddle-github-action.cdn.bcebos.com/windows/tp_predownload/onnxruntime-win-x64-1.11.1.zip')"
if not exist "third_party/onnxruntime/Windows" mkdir "third_party/onnxruntime/Windows"
move onnxruntime-win-x64-1.11.1.zip third_party/onnxruntime/Windows/1.11.1.zip

python -c "import wget;wget.download('https://paddle-github-action.cdn.bcebos.com/windows/tp_predownload/paddle2onnx-win-x64-1.0.0rc2.zip')"
if not exist "third_party/paddle2onnx/Windows" mkdir "third_party/paddle2onnx/Windows"
move paddle2onnx-win-x64-1.0.0rc2.zip third_party/paddle2onnx/Windows/1.0.0rc2.zip

python -c "import wget;wget.download('https://paddle-github-action.cdn.bcebos.com/windows/tp_predownload/dirent-1.23.2.tar.gz')"
if not exist "third_party/dirent" mkdir "third_party/dirent"
move dirent-1.23.2.tar.gz third_party/dirent/1.23.2.tar.gz
