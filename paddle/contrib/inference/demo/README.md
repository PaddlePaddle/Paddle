# Infernce Demos

## MobileNet
Input data format:

- Each line contains a single record
- Each record's format is

```
<space splitted floats as data>\t<space splitted ints as shape>
```

Follow the C++ codes in `mobilenet.cc`.

To execute the demo, simply run

```sh
./mobilenet_inference_demo --modeldir <model> --data <datafile>
```
