# define feeding map
feeding = {'x': 0, 'y': 1}

trainer.train(
    reader=paddle.batch(
        train_reader(file_name), batch_size=10),
    feeding=feeding,
    event_handler=event_handler,
    num_passes=10)
