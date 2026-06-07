FROM ubuntu:22.04

LABEL maintainer="Yin & Zhang"
LABEL description="PEBI Black-Oil Shale Reservoir Simulator v1.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential gfortran cmake git wget curl \
    python3 python3-pip python3-dev \
    libopenmpi-dev openmpi-bin \
    libblas-dev liblapack-dev libhdf5-dev \
    && apt-get clean

RUN pip3 install --no-cache-dir \
    numpy==1.24.3 scipy==1.10.1 matplotlib==3.7.1 \
    pyyaml==6.0 mpi4py==3.1.4 h5py==3.8.0 \
    petsc4py==3.19.0 slepc4py==3.19.0

ENV PETSC_DIR=/opt/petsc
ENV PETSC_ARCH=arch-linux-c-opt
RUN mkdir -p /opt && cd /opt && \
    git clone -b v3.19.0 https://gitlab.com/petsc/petsc.git petsc && \
    cd petsc && \
    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
        --download-parmetis --download-metis --download-fblaslapack \
        --with-debugging=0 \
        COPTFLAGS="-O3" CXXOPTFLAGS="-O3" FOPTFLAGS="-O3" && \
    make all

ENV SLEPC_DIR=/opt/slepc
RUN cd /opt && \
    git clone -b v3.19.0 https://gitlab.com/slepc/slepc.git slepc && \
    cd slepc && ./configure && make all

WORKDIR /app
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY spe10/ /app/spe10/

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${SLEPC_DIR}/${PETSC_ARCH}/lib:${LD_LIBRARY_PATH}"

ENTRYPOINT ["python3", "scripts/run_simulation.py"]
CMD ["--help"]