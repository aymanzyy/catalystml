export MPICH_GPU_SUPPORT_ENABLED=1
export CATALYST_IMPLEMENTATION_PATHS=/lus/eagle/projects/multiphysics_aesp/azy4/finParaViewCatalyst2/paraview_install/lib64/catalyst
export CATALYST_IMPLEMENTATION_NAME=paraview
export PYTHONPATH="/lus/eagle/projects/multiphysics_aesp/azy4/Catalyst_W_Python/catalyst-install/lib64/python3.11/site-packages/:$PYTHONPATH"

export LD_LIBRARY_PATH="/lus/eagle/projects/multiphysics_aesp/azy4/Catalyst_W_Python/catalyst-install/lib64/:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.1.0.70/lib:$LD_LIBRARY_PATH"
module load PrgEnv-nvhpc/8.5.0
module load craype-accel-nvidia80
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29401

cd /lus/grand/projects/visualization/azy4/catalyst_based_ml/Mini-Apps/ 

mpiexec -n 4 --ppn 4 --depth=1 --cpu-bind depth ./set_affinity_gpu_polaris.sh ./test_build/lbm-proxy-app proxy_input_file.txt ./bridge/autoencoder_train.py