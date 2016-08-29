# DataProvider Tutorial #

DataProvider is responsible for data management in PaddlePaddle, corresponding to <a href = "../trainer_config_helpers_api.html#trainer_config_helpers.layers.data_layer">Data Layer</a>.

## Input Data Format ##
PaddlePaddle uses **Slot** to describe the data layer of neural network. One slot describes one data layer. Each slot stores a series of samples, and each sample contains a set of features. There are three attributes of a slot: 
+ **Dimension**: dimenstion of features
+ **SlotType**: there are 5 different slot types in PaddlePaddle, following table compares the four commonly used ones.

<table border="2" frame="border">
<thead>
<tr>
<th scope="col" class="left">SlotType</th>
<th scope="col" class="left">Feature Description</th>
<th scope="col" class="left">Vector Description</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left"><b>DenseSlot</b></td>
<td class="left">Continuous Features</td>
<td class="left">Dense Vector</td>
</tr>

<tr>
<td class="left"><b>SparseNonValueSlot<b></td>
<td class="left">Discrete Features without weights</td>
<td class="left">Sparse Vector with all non-zero elements equaled to 1</td>
</tr>

<tr>
<td class="left"><b>SparseValueSlot</b></td>
<td class="left">Discrete Features with weights</td>
<td class="left">Sparse Vector</td>
</tr>

<tr>
<td class="left"><b>IndexSlot</b></td>
<td class="left">mostly the same as SparseNonValueSlot, but especially for a single label</td>
<td class="left">Sparse Vector with only one value in each time step</td>
</tr>
</tbody>
</table>
</br>

And the remained one is **StringSlot**. It stores Character String, and can be used for debug or to describe data Id for prediction, etc.
+ **SeqType**: a **sequence** is a sample whose features are expanded in time scale. And a **sub-sequence** is a continous ordered subset of a sequence. For example, (a1, a2) and (a3, a4, a5) are two sub-sequences of one sequence (a1, a2, a3, a4, a5). Following are 3 different sequence types in PaddlePaddle:
  - **NonSeq**: input sample is not sequence
  - **Seq**: input sample is a sequence without sub-sequence
  - **SubSeq**: input sample is a sequence with sub-sequence

## Python DataProvider
  
PyDataProviderWrapper is a python decorator in PaddlePaddle, used to read custom python DataProvider class. It currently supports all SlotTypes and SeqTypes of input data. User should only concern how to read samples from file. Feel easy with its [Use Case](python_case.md) and <a href = "../py_data_provider_wrapper_api.html">API Reference</a>.
