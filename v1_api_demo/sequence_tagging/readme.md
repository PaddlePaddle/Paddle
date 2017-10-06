# Sequence Tagging

This demo is a sequence model for assigning tags to each token in a sentence. The task is described at <a href = "http://www.cnts.ua.ac.be/conll2000/chunking">CONLL2000 Text Chunking</a> task.

## Download data
```bash
cd demo/sequence_tagging
./data/get_data.sh
```

## Train model
```bash
cd demo/sequence_tagging
./train.sh
```

## Model description

We provide two models. One is a linear CRF model (linear_crf.py) with is equivalent to the one at <a href="http://leon.bottou.org/projects/sgd#stochastic_gradient_crfs">leon.bottou.org/projects/sgd</a>. The second one is a stacked bidirectional RNN and CRF model (rnn_crf.py).
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<th scope="col" class="left">Model name</th>
<th scope="col" class="left">Number of parameters</th>
<th scope="col" class="left">F1 score</th>
</thead>

<tbody>
<tr>
<td class="left">linear_crf</td>
<td class="left"> 1.8M </td>
<td class="left"> 0.937</td>
</tr>

<tr>
<td class="left">rnn_crf</td>
<td class="left"> 960K </td>
<td class="left">0.941</td>
</tr>

</tbody>
</table>
</center>
<br>
