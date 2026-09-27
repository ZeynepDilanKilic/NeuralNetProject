#include "neuralnet/Layer.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace neuralnet {

namespace {

/// Half-width of the uniform distribution the weights are drawn from.
///
/// Glorot & Bengio (2010): U(-l, l), l = sqrt(6 / (fan_in + fan_out)) for sigmoid/tanh.
/// He et al. (2015):       U(-l, l), l = sqrt(6 / fan_in)            for the ReLU family.
double initializationLimit(ActivationType type, std::size_t fanIn, std::size_t fanOut) {
    switch (type) {
    case ActivationType::Sigmoid:
    case ActivationType::Tanh:
        return std::sqrt(6.0 / static_cast<double>(fanIn + fanOut));
    case ActivationType::ReLU:
    case ActivationType::LeakyReLU:
        return std::sqrt(6.0 / static_cast<double>(fanIn));
    }
    throw std::invalid_argument("Unknown activation function type.");
}

} // namespace

Layer::Layer(std::size_t numberOfNeurons, std::size_t numberOfInputsPerNeuron,
             ActivationType activationType, std::mt19937& rng)
    : activationType_(activationType) {
    if (numberOfNeurons == 0) {
        throw std::invalid_argument("A layer needs at least one neuron.");
    }
    if (numberOfInputsPerNeuron == 0) {
        throw std::invalid_argument("A layer needs at least one input per neuron.");
    }

    const double limit =
        initializationLimit(activationType, numberOfInputsPerNeuron, numberOfNeurons);
    // NOTE: std::uniform_real_distribution is implementation-defined, so a given seed is
    // reproducible per standard library (libstdc++, libc++, MSVC STL) but not across them.
    std::uniform_real_distribution<double> distribution(-limit, limit);

    neurons_.reserve(numberOfNeurons);
    for (std::size_t n = 0; n < numberOfNeurons; ++n) {
        std::vector<double> weights(numberOfInputsPerNeuron);
        std::ranges::generate(weights, [&] { return distribution(rng); });
        neurons_.emplace_back(std::move(weights), 0.0, activationType);
    }
}

std::vector<double> Layer::processInputs(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(neurons_.size());
    std::ranges::transform(neurons_, std::back_inserter(outputs),
                           [&inputs](Neuron& neuron) { return neuron.output(inputs); });
    return outputs;
}

void Layer::updateWeights(const std::vector<double>& errors, double learningRate,
                          const std::vector<double>& inputs) {
    if (errors.size() != neurons_.size()) {
        throw std::invalid_argument("Error count must match the number of neurons.");
    }
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i].updateParameters(inputs, errors[i], learningRate);
    }
}

std::size_t Layer::getNumberOfInputs() const noexcept {
    return neurons_.front().getNumberOfInputs();
}

const Neuron& Layer::getNeuron(std::size_t index) const {
    return neurons_.at(index);
}

Neuron& Layer::getNeuron(std::size_t index) {
    return neurons_.at(index);
}

} // namespace neuralnet
