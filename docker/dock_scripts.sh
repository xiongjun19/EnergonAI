docker build -t energon -f Dockerfile.torch .
cnt_name='cnt_energon_xj'
loc_data='/data/energon'
docker run --name ${cnt_name} -it --gpus all --ipc=host -v `pwd`:/workspace/energon -v ${loc_data}:/workspace/data energon 
