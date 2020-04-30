#!/bin/bash

# download baseline model file to ./model_baseline/
if [ -d ./model_baseline/ ]
then
    echo "./model_baseline/ directory already existed, ignore download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/sequence_tagging_dy.tar.gz
    tar xvf sequence_tagging_dy.tar.gz
    /bin/rm sequence_tagging_dy.tar.gz
fi

# download dataset file to ./data/
if [ -d ./data/ ]
then
    echo "./data/ directory already existed, ignore download"
else
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-dataset-2.0.0.tar.gz
    tar xvf lexical_analysis-dataset-2.0.0.tar.gz
    /bin/rm lexical_analysis-dataset-2.0.0.tar.gz
fi

