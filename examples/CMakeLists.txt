add_executable(training_check training_check.cpp)
target_link_libraries(training_check PRIVATE neuralnet::neuralnet neuralnet::warnings)

add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE neuralnet::neuralnet neuralnet::warnings)

# Both examples return non-zero when they fail, so registering them with CTest makes them
# part of every `ctest` run (including the sanitizer and coverage jobs).
add_test(NAME example.training_check COMMAND training_check)
add_test(NAME example.xor COMMAND xor)
