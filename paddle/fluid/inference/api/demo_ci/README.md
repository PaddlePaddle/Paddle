# Inference Demos

There are several demos:

- simple_on_word2vec: 
  - Follow the C++ codes is in `simple_on_word2vec.cc`. 
  - It is suitable for word2vec model.
- vis_demo: 
  - Follow the C++ codes is in `vis_demo.cc`. 
  - It is suitable for mobilenet, se_resnext50 and ocr three models.
  - Input data format:
    - Each line contains a single record
    - Each record's format is
    ```
    <space splitted floats as data>\t<space splitted ints as shape>
    ```
- infer_image_classification:
  - The C++ code is in `infer_image_classification.cc`.
  - It accepts e.g. ResNet50, SE-ResNeXt50 and MobileNet-v1 models.
  - Requires `ImageNet` directory with the ImageNet dataset

To build and execute the demos, simply run 
```
./run.sh $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU $DATA_DIR
```
- It will build and execute the demos in both static and shared library.
- `$PADDLE_ROOT`:  paddle library path
- `$TURN_ON_MKL`:  use MKL or Openblas
- `$TEST_GPU_CPU`: test both GPU/CPU mode or only CPU mode
- `$DATA_DIR`:     a directory to store models and datasets in
- NOTE: for simple_on_word2vec, must run `ctest -R test_word2vec -R` to obtain word2vec model at first.

To build only a single demo, run
```
./build.sh <demo_name> $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU $WITH_STATIC_LIB
```
where `$WITH_STATIC_LIB` determines the linkage (static/dynamic) to PaddlePaddle fluid library.
