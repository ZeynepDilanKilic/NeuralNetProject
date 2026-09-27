#include "neuralnet/Neuron.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace neuralnet {

Neuron::Neuron(std::vector<double> weights, double bias, ActivationType activationType)
    : weights_(std::move(weights)), bias_(bias), activationType_(activationType) {
    if (weights_.empty()) {
        throw std::invalid_argument("A neuron needs at least one weight.");
    }
}

double Neuron::output(const std::vector<double>& inputs) {
    if (inputs.size() != weights_.size()) {
        throw std::invalid_argument("Input size must match the number of weights.");
    }

    double preActivation = bias_;
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        preActivation += weights_[i] * inputs[i];
    }

    lastOutput_ = activate(activationType_, preActivation);
    return lastOutput_;
}

void Neuron::updateParameters(const std::vector<double>& inputs, double error,
                              double learningRate) {
    if (inputs.size() != weights_.size()) {
        throw std::invalid_argument("Input size must match the number of weights.");
    }

    // error = dL/da, delta = dL/dz = dL/da * da/dz  (z = w . x + b, a = f(z))
    const double delta = error * activationDerivative(activationType_, lastOutput_);

    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learningRate * delta * inputs[i];
    }
    bias_ -= learningRate * delta;
}

double Neuron::activate(ActivationType type, double x) {
    switch (type) {
    case ActivationType::Sigmoid:
        return 1.0 / (1.0 + std::exp(-x));
    case ActivationType::ReLU:
        return x > 0.0 ? x : 0.0;
    case ActivationType::Tanh:
        // std::tanh is overflow-safe; (e^x - e^-x) / (e^x + e^-x) is not for |x| > ~709.
        return std::tanh(x);
    case ActivationType::LeakyReLU:
        return x > 0.0 ? x : kLeakyReluSlope * x;
    }
    throw std::invalid_argument("Unknown activation function type.");
}

double Neuron::activationDerivative(ActivationType type, double output) {
    switch (type) {
    case ActivationType::Sigmoid:
        return output * (1.0 - output);
    case ActivationType::ReLU:
        return output > 0.0 ? 1.0 : 0.0;
    case ActivationType::Tanh:
        return 1.0 - (output * output);
    case ActivationType::LeakyReLU:
        return output > 0.0 ? 1.0 : kLeakyReluSlope;
    }
    throw std::invalid_argument("Unknown activation function type.");
}

double Neuron::getWeight(std::size_t index) const {
    return weights_.at(index);
}

void Neuron::setWeight(std::size_t index, double value) {
    weights_.at(index) = value;
}

} // namespace neuralnet
