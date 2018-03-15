本目录下的文件是实现美团外卖送达时间预测任务所需要的原始数据文件和代码文件。
1. 文件说明
	all_data目录：该目录下包含所有原始数据.
	features_project.py:	该文件主要功能是从all_data目录下的所有原始数据文件中提取用于任务的特征数据，
							并且将提取的特征存储到本地文件中.	

	feature_data.csv:		该文件是已经通过features_porjcet.py文件提出并处理得到的特征数据文件.
	predict_model.py:		该文件主要功能是根据特征数据文件feature_data.csv训练模型，并且通过训练好的模型预测测试数据集，
							并将预测结果存储到本地文件中.

2. 使用说明
	#根据原始数据生成特征数据文件,已经默认生成了特征数据文件feature_data.csv
	./features_project.py all_data/ feature_data.csv

	#使用特征数据训练模型并预测测试数据集，存储测试数据集的预测结果到本地文件
	#参数1表示需要输入的特征数据文件；参数2表示预测结果文件路径
	./predict_model.py  feature_data.csv  evaluate_result.csv
