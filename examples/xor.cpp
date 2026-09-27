// Trains a small network on XOR and prints the learned truth table.
#include "neuralnet/NeuralNetwork.h"

#include <array>
#include <iomanip>
#include <iostream>

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
    NeuralNetwork network({2, 4, 1}, ActivationType::Tanh);

    std::cout << std::fixed << std::setprecision(4);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epochLoss = 0.0;
        for (const Sample& sample : dataset) {
            network.train(sample.inputs, sample.target, learningRate);
            epochLoss += network.computeLoss(sample.inputs, sample.target);
        }
        if (epoch % 1000 == 0) {
            std::cout << "epoch " << std::setw(5) << epoch << "  loss " << epochLoss << '\n';
        }
    }

    std::cout << "\n a  b | a XOR b\n---------------\n";
    for (const Sample& sample : dataset) {
        std::cout << ' ' << sample.inputs[0] << ' ' << sample.inputs[1] << " | "
                  << network.predict(sample.inputs)[0] << '\n';
    }
    return 0;
}
