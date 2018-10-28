apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# Under Ubuntu 16.04 the CMake configuration scripts of CGAL
# are broken and you must also install the CGAL Qt5 package:
apt-get install -y libcgal-qt5-dev

# Install Ceres Solver:
apt-get install -y libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make
make install

# Configure and compile COLMAP:
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake ..
make
make install

cd /usr/local/lib/colmap
for staticLib in ./*.a; do
  g++ -shared -o $(basename $staticLib .a).so -Wl,--whole-archive $staticLib -Wl,--no-whole-archive
done
git clone https://github.com/garyjyzhang/scanner-colmap.git
cd scanner-colmap/integration/op_cpp
mkdir build
cd build
cmake ..
make
cd ../..
