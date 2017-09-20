word_dict = dict()
...  #  read dictionary from outside

# define feeding map
feeding = {'x': 0, 'y': 1}

trainer.train(
        reader=paddle.batch(
            train_reader(file_name,word_dict), batch_size=10),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=10)