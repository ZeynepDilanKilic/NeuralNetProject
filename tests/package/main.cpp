// Links against an installed NeuralNet and checks that one training step lowers the loss.
#include "neuralnet/NeuralNetwork.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using neuralnet::ActivationType;
    using neuralnet::NeuralNetwork;

    NeuralNetwork network(2, {{3, ActivationType::Tanh}, {1, ActivationType::Linear}});
    const std::vector<double> inputs{0.5, -0.25};
    const std::vector<double> targets{0.75};

    const double lossBefore = network.train(inputs, targets, 0.1);
    const double lossAfter = network.computeLoss(inputs, targets);
    if (!std::isfinite(lossAfter) || lossAfter >= lossBefore) {
        std::cerr << "Installed NeuralNet package does not train: " << lossBefore << " -> "
                  << lossAfter << '\n';
        return 1;
    }
    std::cout << "Installed NeuralNet package works: loss " << lossBefore << " -> " << lossAfter
              << '\n';
    return 0;
}
