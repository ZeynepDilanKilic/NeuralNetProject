// Smallest possible sanity check: one SGD step on a single neuron must reduce the loss.
#include "neuralnet/NeuralNetwork.h"

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    using neuralnet::ActivationType;
    using neuralnet::NeuralNetwork;

    NeuralNetwork network({1}, ActivationType::Sigmoid);
    const std::vector<double> inputs{1.0};
    const std::vector<double> targets{1.0};

    const double lossBefore = network.computeLoss(inputs, targets);
    const double before = network.predict(inputs).at(0);
    network.train(inputs, targets, 0.1);
    const double lossAfter = network.computeLoss(inputs, targets);
    const double after = network.predict(inputs).at(0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Prediction: " << before << " -> " << after << '\n';
    std::cout << "Loss:       " << lossBefore << " -> " << lossAfter << '\n';

    if (!std::isfinite(lossAfter) || lossAfter >= lossBefore) {
        std::cerr << "Training check failed.\n";
        return 1;
    }
    std::cout << "Training check passed.\n";
    return 0;
}
