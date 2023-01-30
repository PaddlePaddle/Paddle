# Inference Demos

There are several demos:

<<<<<<< HEAD
- simple_on_word2vec:
  - Follow the C++ codes is in `simple_on_word2vec.cc`.
  - It is suitable for word2vec model.
- vis_demo:
  - Follow the C++ codes is in `vis_demo.cc`.
=======
- simple_on_word2vec: 
  - Follow the C++ codes is in `simple_on_word2vec.cc`. 
  - It is suitable for word2vec model.
- vis_demo: 
  - Follow the C++ codes is in `vis_demo.cc`. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  - It is suitable for mobilenet, se_resnext50 and ocr three models.
  - Input data format:
    - Each line contains a single record
    - Each record's format is
    ```
    <space split floats as data>\t<space split ints as shape>
    ```

<<<<<<< HEAD
To build and execute the demos, simply run
=======
To build and execute the demos, simply run 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
```
./run.sh $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU
```
- It will build and execute the demos in both static and shared library.
- `$PADDLE_ROOT`: paddle library path
- `$TURN_ON_MKL`: use MKL or Openblas
- `$TEST_GPU_CPU`: test both GPU/CPU mode or only CPU mode
- NOTE: for simple_on_word2vec, must run `ctest -R test_word2vec -R` to obtain word2vec model at first.
