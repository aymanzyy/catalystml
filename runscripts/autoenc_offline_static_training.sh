export MPICH_GPU_SUPPORT_ENABLED=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29401

mpiexec -n 4 --ppn 4 --depth=1 --cpu-bind depth ./set_affinity_gpu_polaris.sh python3 ./models/offline_static_training.py
