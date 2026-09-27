# Prefer an installed Catch2 (e.g. via vcpkg or the system package manager);
# otherwise fetch it so a fresh clone builds with nothing but CMake and a compiler.
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

add_executable(neuralnet_tests
    activation_test.cpp
    gradient_check_test.cpp
    training_test.cpp
    validation_test.cpp)
target_link_libraries(neuralnet_tests PRIVATE
    neuralnet::neuralnet
    neuralnet::warnings
    Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(neuralnet_tests)
