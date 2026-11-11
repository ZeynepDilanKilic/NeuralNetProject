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
option(NEURALNET_INSTALL           "Generate install rules and a CMake package" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Only pick a default build type when this is the top-level project; as a subproject the
# parent owns CMAKE_BUILD_TYPE and must not find it forced into its cache.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
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
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(neuralnet PUBLIC cxx_std_20)
# BUILD_INTERFACE keeps the build-only warnings target out of the static library's exported
# link interface; without it install(EXPORT) fails because neuralnet::warnings is not exported.
target_link_libraries(neuralnet PRIVATE $<BUILD_INTERFACE:neuralnet::warnings>)

# ---------------------------------------------------------------------------
# Installation and CMake package (find_package(NeuralNet) + neuralnet::neuralnet)
# ---------------------------------------------------------------------------
if(NEURALNET_INSTALL)
    include(CMakePackageConfigHelpers)
    set(NEURALNET_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNet)

    install(TARGETS neuralnet
        EXPORT NeuralNetTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT NeuralNetTargets
        NAMESPACE neuralnet::
        DESTINATION ${NEURALNET_CMAKE_DIR})

    configure_package_config_file(cmake/NeuralNetConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfig.cmake
        INSTALL_DESTINATION ${NEURALNET_CMAKE_DIR})
    # 0.x releases may break the API on every minor version, so only the same minor matches.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfigVersion.cmake
        DESTINATION ${NEURALNET_CMAKE_DIR})
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/neuralnet)
endif()

# ---------------------------------------------------------------------------
# Examples and tests
# ---------------------------------------------------------------------------
# The examples double as CTest tests, so testing is enabled whenever either is built. This
# only affects this directory tree, so it is also correct when NeuralNet is a subproject.
if(NEURALNET_BUILD_TESTS OR NEURALNET_BUILD_EXAMPLES)
    enable_testing()
endif()

if(NEURALNET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NEURALNET_BUILD_TESTS)
    add_subdirectory(tests)
endif()
