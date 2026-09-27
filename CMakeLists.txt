cmake_minimum_required(VERSION 3.21)

project(NeuralNet
    VERSION 0.1.0
    DESCRIPTION "Dependency-free C++20 feed-forward neural network library"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NEURALNET_BUILD_TESTS       "Build the unit tests"                       ${PROJECT_IS_TOP_LEVEL})
option(NEURALNET_BUILD_EXAMPLES    "Build the example programs"                 ${PROJECT_IS_TOP_LEVEL})
option(NEURALNET_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"         ${PROJECT_IS_TOP_LEVEL})
option(NEURALNET_ENABLE_SANITIZERS "Build with AddressSanitizer + UBSan"        OFF)
option(NEURALNET_ENABLE_COVERAGE   "Instrument for line coverage (GCC/Clang)"   OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/CompilerWarnings.cmake)

if(NEURALNET_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "Sanitizers are only configured for GCC/Clang; ignoring NEURALNET_ENABLE_SANITIZERS.")
    else()
        set(NEURALNET_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
        add_compile_options(${NEURALNET_SANITIZER_FLAGS})
        add_link_options(${NEURALNET_SANITIZER_FLAGS})
    endif()
endif()

if(NEURALNET_ENABLE_COVERAGE)
    if(MSVC)
        message(WARNING "Coverage is only configured for GCC/Clang; ignoring NEURALNET_ENABLE_COVERAGE.")
    else()
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
    endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(neuralnet STATIC
    src/Neuron.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp)
add_library(neuralnet::neuralnet ALIAS neuralnet)

target_include_directories(neuralnet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(neuralnet PUBLIC cxx_std_20)
target_link_libraries(neuralnet PRIVATE neuralnet::warnings)

# ---------------------------------------------------------------------------
# Examples and tests
# ---------------------------------------------------------------------------
if(NEURALNET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NEURALNET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
