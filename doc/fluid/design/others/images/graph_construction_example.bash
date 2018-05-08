cat ./graph_construction_example.dot | \
    sed 's/color=red/color=red, style=invis/g' | \
    sed 's/color=green/color=green, style=invis/g' | \
    dot -Tpng > graph_construction_example_forward_only.png

cat ./graph_construction_example.dot | \
    sed 's/color=green/color=green, style=invis/g' | \
    dot -Tpng > graph_construction_example_forward_backward.png

cat ./graph_construction_example.dot | \
    dot -Tpng > graph_construction_example_all.png
