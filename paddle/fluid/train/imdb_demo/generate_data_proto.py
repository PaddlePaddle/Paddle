
import paddle.fluid as fluid

dataset = fluid.DatasetFactory().create_dataset()
dataset.set_batch_size(128)
dataset.set_pipe_command("python imdb_reader.py")

data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

dataset.set_use_var([data, label])
desc = dataset.proto_desc

with open("data.proto", "w") as f:
    f.write(dataset.desc())
