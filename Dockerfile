FROM nvidia/cuda

RUN apt-get update -y && \
    apt-get install -y \
        file \
        hwloc \
        openssh-client \
        libibverbs-dev \
        libboost-all-dev \
        wget \
        cmake \
        git

RUN mkdir -p /tmp && wget -q --no-check-certificate -P /tmp https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2 && \
    tar -x -f /tmp/openmpi-3.0.0.tar.bz2 -C /tmp -j && \
    cd /tmp/openmpi-3.0.0 && ./configure --prefix=/usr/local/openmpi --disable-getpwuid --enable-orterun-prefix-by-default --with-cuda --with-verbs && \
    make -j4 && \
    make -j4 install && \
    rm -rf /tmp/openmpi-3.0.0.tar.bz2 /tmp/openmpi-3.0.0

ENV PATH=/usr/local/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

RUN wget -q --no-check-certificate -P /tmp https://arrayfire.s3.amazonaws.com/3.7.1/ArrayFire-v3.7.1-1_Linux_x86_64.sh && \
    chmod +x /tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh && \
    ./tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh --include-subdir --prefix=/opt

ENV LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH

RUN wget -q --no-check-certificate -P /tmp https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/CMake-hdf5-1.10.6.tar.gz && \
    tar -x -f /tmp/CMake-hdf5-1.10.6.tar.gz -C /tmp && \
    cd /tmp/CMake-hdf5-1.10.6/hdf5-1.10.6 && \
    mkdir build && \
    cd build && \
    CC=gcc CXX=g++ cmake -DHDF5_ENABLE_PARALLEL=ON -DHDF5_BUILD_CPP_LIB=OFF -DCMAKE_INSTALL_PREFIX=/opt/hdf5 .. && \
    make install

RUN git clone --recursive https://github.com/keichi/mpEDM.git && \
    cd mpEDM && \
    mkdir build && \
    cd build && \
    cmake -DHDF5_DIR=/opt/hdf5/share/cmake/hdf5/ -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake/ -DCMAKE_BUILD_TYPE=Release .. && \
    make

ENV PATH=/mpEDM/build:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH