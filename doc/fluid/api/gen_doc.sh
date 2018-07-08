#!/bin/bash
python gen_doc.py layers --submodules control_flow device io nn ops tensor learning_rate_scheduler detection metric_op tensor > layers.rst

for module in data_feeder clip metrics executor initializer io nets optimizer param_attr profiler regularizer transpiler recordio_writer backward average profiler
do
  python gen_doc.py ${module} > ${module}.rst
done

python gen_doc.py "" > fluid.rst
