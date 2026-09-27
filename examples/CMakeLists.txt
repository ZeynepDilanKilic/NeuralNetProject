add_executable(training_check training_check.cpp)
target_link_libraries(training_check PRIVATE neuralnet::neuralnet neuralnet::warnings)

add_executable(xor xor.cpp)
target_link_libraries(xor PRIVATE neuralnet::neuralnet neuralnet::warnings)
