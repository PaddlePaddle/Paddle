#!/bin/bash

library_path=$1
mkldnn_lib=$library_path"/third_party/install/mkldnn/lib"
mklml_lib=$library_path"/third_party/install/mklml/lib"
paddle_inference_lib=$library_path"/paddle/lib"
export LD_LIBRARY_PATH=$mkldnn_lib:$mklml_lib:$paddle_inference_lib:.
javac -cp $CLASSPATH:JavaInference.jar:. test.java
java -cp $CLASSPATH:JavaInference.jar:. test $2 $3
