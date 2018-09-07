#!/bin/bash
python gen_doc.py layers --submodules control_flow device io nn ops tensor > layers.rst

for module in data_feeder clip metrics executor initializer io nets optimizer param_attr profiler regularizer
do
  python gen_doc.py ${module} > ${module}.rst
done
