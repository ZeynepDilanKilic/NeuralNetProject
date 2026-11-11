# Stand-alone consumer of an *installed* NeuralNet, built by the CI `package` job:
#
#   cmake --install build --prefix stage
#   cmake -S tests/package -B build-consumer -DCMAKE_PREFIX_PATH=$PWD/stage
#   cmake --build build-consumer && ctest --test-dir build-consumer
#
# It proves that find_package(NeuralNet) and the neuralnet::neuralnet target work outside
# the source tree. It is deliberately not part of the main build.
cmake_minimum_required(VERSION 3.21)
project(NeuralNetConsumer LANGUAGES CXX)

find_package(NeuralNet REQUIRED CONFIG)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE neuralnet::neuralnet)

enable_testing()
add_test(NAME consumer COMMAND consumer)
