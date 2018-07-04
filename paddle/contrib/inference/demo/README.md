# Infernce Demos

Input data format:

- Each line contains a single record
- Each record's format is

```
<space splitted floats as data>\t<space splitted ints as shape>
```

Follow the C++ codes in `vis_demo.cc`.

## MobileNet

To execute the demo, simply run

```sh
./mobilenet_inference_demo --modeldir <model> --data <datafile>
```

## SE-ResNeXt-50

To execute the demo, simply run

```sh
./se_resnext50_inference_demo --modeldir <model> --data <datafile>
```

## OCR

To execute the demo, simply run

```sh
./ocr_inference_demo --modeldir <model> --data <datafile>
```
