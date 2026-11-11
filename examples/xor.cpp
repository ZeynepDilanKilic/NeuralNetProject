// Trains a small network on XOR, prints the learned truth table and exits non-zero if any
// prediction is more than 0.1 away from its target (the example is also a CTest test).
#include "neuralnet/NeuralNetwork.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using neuralnet::ActivationType;
    using neuralnet::NeuralNetwork;

    struct Sample {
        std::vector<double> inputs;
        std::vector<double> target;
    };
    const std::array<Sample, 4> dataset{{
        {.inputs = {0.0, 0.0}, .target = {0.0}},
        {.inputs = {0.0, 1.0}, .target = {1.0}},
        {.inputs = {1.0, 0.0}, .target = {1.0}},
        {.inputs = {1.0, 1.0}, .target = {0.0}},
    }};

    constexpr int epochs = 5000;
    constexpr double learningRate = 0.5;
    constexpr double tolerance = 0.1;
    NeuralNetwork network({2, 4, 1}, ActivationType::Tanh);

    std::cout << std::fixed << std::setprecision(4);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epochLoss = 0.0;
        for (const Sample& sample : dataset) {
            // train() returns the sample's loss before the update, so no extra forward pass.
            epochLoss += network.train(sample.inputs, sample.target, learningRate);
        }
        if (epoch % 1000 == 0) {
            std::cout << "epoch " << std::setw(5) << epoch << "  loss " << epochLoss << '\n';
        }
    }

    bool learned = true;
    std::cout << "\n a  b | a XOR b\n---------------\n";
    for (const Sample& sample : dataset) {
        const double prediction = network.predict(sample.inputs).at(0);
        std::cout << ' ' << sample.inputs[0] << ' ' << sample.inputs[1] << " | " << prediction
                  << '\n';
        learned = learned && std::abs(prediction - sample.target[0]) < tolerance;
    }

    if (!learned) {
        std::cerr << "XOR was not learned to within " << tolerance << ".\n";
        return 1;
    }
    return 0;
}
