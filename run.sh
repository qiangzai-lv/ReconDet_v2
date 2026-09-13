bash tools/dist_train.sh configs/recondet/recondet_arkit.py 1

/root/miniforge3/bin/python \
  utils/benchmark_cuda_npu_geometry.py \
  --device cuda \
  --warmup 10 \
  --repeats 50 \
  --output ./cuda_geometry.json

/root/miniforge3/bin/python \
  utils/benchmark_cuda_npu_geometry.py \
  --device npu \
  --warmup 10 \
  --repeats 50 \
  --output npu_geometry.json

/root/miniforge3/bin/python \
  utils/benchmark_cuda_npu_geometry.py \
  --compare cuda_geometry.json npu_geometry.json \
  --atol 1e-4 \
  --rtol 1e-4 \
  --json

/root/miniforge3/bin/python \
  utils/benchmark_cuda_npu_geometry.py \
  --compare cuda_geometry.json npu_geometry.json \
  --strict

/root/miniconda3/envs/mmdet/bin/python \
  utils/diagnose_npu_runtime.py \
  --json

python -c "import torch, torch_npu, mx_driving; \
print('torch=', torch.__version__); \
print('torch_npu=', getattr(torch_npu, '__version__', 'unknown')); \
print('mx_driving=', mx_driving.__file__)"

