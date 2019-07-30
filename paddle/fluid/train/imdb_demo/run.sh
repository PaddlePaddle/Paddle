
set -exu
build/demo_trainer \
	train_filelist.txt \
	data.proto \
        startup_program \
        main_program


