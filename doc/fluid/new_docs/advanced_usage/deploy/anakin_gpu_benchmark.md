# Anakin GPU Benchmark

## Machine:

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`
>  cuDNN: `v7`


## Counterpart of anakin  :

The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 3`** ,   The models which TensorRT 3 doesn't support we use the custom plugins  to support.

## Benchmark Model

The following convolutional neural networks are tested with both `Anakin` and `TenorRT3`.
 You can use pretrained caffe model or the model trained by youself.

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](../docs/Manual/Converter_en.md)


- [Vgg16](#1)   *caffe model can be found [here->](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)*
- [Yolo](#2)  *caffe model can be found [here->](https://github.com/hojel/caffe-yolo-model)*
- [Resnet50](#3)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Resnet101](#4)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Mobilenet v1](#5)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2](#6)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [RNN](#7)  *not support yet*

We tested them on single-GPU with single-thread.

### <span id = '1'>VGG16 </span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>8.8690</td>
        <td>8.2815</td>
      </tr>
      <tr>
        <td>2</td>
        <td>15.5344</td>
        <td>13.9116</td>
      </tr>
      <tr>
        <td>4</td>
        <td>26.6000</td>
        <td>21.8747</td>
      </tr>
      <tr>
        <td>8</td>
        <td>49.8279</td>
        <td>40.4076</td>
      </tr>
      <tr>
        <td>32</td>
        <td>188.6270</td>
        <td>163.7660</td>
      </tr>
    </tbody>
  </table>
</p>

- GPU Memory Used (`MB`)

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>963</td>
        <td>997</td>
      </tr>
      <tr>
        <td>2</td>
        <td>965</td>
        <td>1039</td>
      </tr>
      <tr>
        <td>4</td>
        <td>991</td>
        <td>1115</td>
      </tr>
      <tr>
        <td>8</td>
        <td>1067</td>
        <td>1269</td>
      </tr>
      <tr>
        <td>32</td>
        <td>1715</td>
        <td>2193</td>
      </tr>
    </tbody>
  </table>
</p>


### <span id = '2'>Yolo </span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>16.4596</td>
        <td>15.2124</td>
      </tr>
      <tr>
        <td>2</td>
        <td>26.6347</td>
        <td>25.0442</td>
      </tr>
      <tr>
        <td>4</td>
        <td>43.3695</td>
        <td>43.5017</td>
      </tr>
      <tr>
        <td>8</td>
        <td>80.9139</td>
        <td>80.9880</td>
      </tr>
      <tr>
        <td>32</td>
        <td>293.8080</td>
        <td>310.8810</td>
      </tr>
    </tbody>
  </table>
</p>


- GPU Memory Used (`MB`)

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>1569</td>
        <td>1775</td>
      </tr>
      <tr>
        <td>2</td>
        <td>1649</td>
        <td>1815</td>
      </tr>
      <tr>
        <td>4</td>
        <td>1709</td>
        <td>1887</td>
      </tr>
      <tr>
        <td>8</td>
        <td>1731</td>
        <td>2031</td>
      </tr>
      <tr>
        <td>32</td>
        <td>2253</td>
        <td>2907</td>
      </tr>
    </tbody>
  </table>
</p>

### <span id = '3'> Resnet50 </span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>4.2459</td>
        <td>4.1061</td>
      </tr>
      <tr>
        <td>2</td>
        <td>6.2627</td>
        <td>6.5159</td>
      </tr>
      <tr>
        <td>4</td>
        <td>10.1277</td>
        <td>11.3327</td>
      </tr>
      <tr>
        <td>8</td>
        <td>17.8209</td>
        <td>20.6680</td>
      </tr>
      <tr>
        <td>32</td>
        <td>65.8582</td>
        <td>77.8858</td>
      </tr>
    </tbody>
  </table>
</p>

- GPU Memory Used (`MB`)

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>531</td>
        <td>503</td>
      </tr>
      <tr>
        <td>2</td>
        <td>543</td>
        <td>517</td>
      </tr>
      <tr>
        <td>4</td>
        <td>583</td>
        <td>541</td>
      </tr>
      <tr>
        <td>8</td>
        <td>611</td>
        <td>589</td>
      </tr>
      <tr>
        <td>32</td>
        <td>809</td>
        <td>879</td>
      </tr>
    </tbody>
  </table>
</p>

### <span id = '4'> Resnet101 </span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>7.5562</td>
        <td>7.0837</td>
      </tr>
      <tr>
        <td>2</td>
        <td>11.6023</td>
        <td>11.4079</td>
      </tr>
      <tr>
        <td>4</td>
        <td>18.3650</td>
        <td>20.0493</td>
      </tr>
      <tr>
        <td>8</td>
        <td>32.7632</td>
        <td>36.0648</td>
      </tr>
      <tr>
        <td>32</td>
        <td>123.2550</td>
        <td>135.4880</td>
      </tr>
    </tbody>
  </table>
</p>

- GPU Memory Used (`MB)`

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>701</td>
        <td>683</td>
      </tr>
      <tr>
        <td>2</td>
        <td>713</td>
        <td>697</td>
      </tr>
      <tr>
        <td>4</td>
        <td>793</td>
        <td>721</td>
      </tr>
      <tr>
        <td>8</td>
        <td>819</td>
        <td>769</td>
      </tr>
      <tr>
        <td>32</td>
        <td>1043</td>
        <td>1059</td>
      </tr>
    </tbody>
  </table>
</p>

###  <span id = '5'> MobileNet V1 </span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>45.5156</td>
        <td>1.3947</td>
      </tr>
      <tr>
        <td>2</td>
        <td>46.5585</td>
        <td>2.5483</td>
      </tr>
      <tr>
        <td>4</td>
        <td>48.4242</td>
        <td>4.3404</td>
      </tr>
      <tr>
        <td>8</td>
        <td>52.7957</td>
        <td>8.1513</td>
      </tr>
      <tr>
        <td>32</td>
        <td>83.2519</td>
        <td>31.3178</td>
      </tr>
    </tbody>
  </table>
</p>
- GPU Memory Used (`MB`)

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>329</td>
        <td>283</td>
      </tr>
      <tr>
        <td>2</td>
        <td>345</td>
        <td>289</td>
      </tr>
      <tr>
        <td>4</td>
        <td>371</td>
        <td>299</td>
      </tr>
      <tr>
        <td>8</td>
        <td>393</td>
        <td>319</td>
      </tr>
      <tr>
        <td>32</td>
        <td>531</td>
        <td>433</td>
      </tr>
    </tbody>
  </table>
</p>


###  <span id = '6'> MobileNet V2</span>

- Latency (`ms`) of different batch

<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>65.6861</td>
        <td>2.9842</td>
      </tr>
      <tr>
        <td>2</td>
        <td>66.6814</td>
        <td>4.7472</td>
      </tr>
      <tr>
        <td>4</td>
        <td>69.7114</td>
        <td>7.4163</td>
      </tr>
      <tr>
        <td>8</td>
        <td>76.1092</td>
        <td>12.8779</td>
      </tr>
      <tr>
        <td>32</td>
        <td>124.9810</td>
        <td>47.2142</td>
      </tr>
    </tbody>
  </table>
</p>

- GPU Memory Used (`MB`)
<p align="center">
  <table>
    <thead>
      <tr>
        <th>BatchSize</th>
        <th>TensorRT</th>
        <th>Anakin</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>341</td>
        <td>293</td>
      </tr>
      <tr>
        <td>2</td>
        <td>353</td>
        <td>301</td>
      </tr>
      <tr>
        <td>4</td>
        <td>385</td>
        <td>319</td>
      </tr>
      <tr>
        <td>8</td>
        <td>421</td>
        <td>351</td>
      </tr>
      <tr>
        <td>32</td>
        <td>637</td>
        <td>551</td>
      </tr>
    </tbody>
  </table>
</p>

## How to run those Benchmark models?

> 1. At first, you should parse the caffe model with [`external converter`](https://github.com/PaddlePaddle/Anakin/blob/b95f31e19993a192e7428b4fcf852b9fe9860e5f/docs/Manual/Converter_en.md).
> 2. Switch to *source_root/benchmark/CNN* directory. Use 'mkdir ./models' to create ./models and put anakin models into this file.
> 3. Use command 'sh run.sh', we will create files in logs to save model log with different batch size. Finally, model latency summary will be displayed on the screen.
> 4. If you want to get more detailed information with op time, you can modify CMakeLists.txt with setting `ENABLE_OP_TIMER` to `YES`, then recompile and run. You will find detailed information in  model log file.
