echo "2 Lines for GPU from TF website" &&
eval "$(conda shell.bash hook)" &&
conda activate tf &&
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) &&
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib &&
$(python -c "from tensorflow.python.client import device_lib;  print(device_lib.list_local_devices());")
