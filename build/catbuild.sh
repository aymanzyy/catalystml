## Clone repository
git clone --branch v2.0.0 https://gitlab.kitware.com/paraview/catalyst.git

## Configure the build
cmake ./Catalyst_Stub/catalyst-v2.0.0/

## Run tests
ctest

## Install the build
cmake --install . --prefix ./Catalyst_Stub/catalyst-install/
