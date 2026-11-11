#include "neuralnet/Layer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace neuralnet {

namespace {

/// Half-width of the uniform distribution the weights are drawn from.
///
/// Glorot & Bengio (2010): U(-l, l), l = sqrt(6 / (fan_in + fan_out)) for sigmoid/tanh/linear.
/// He et al. (2015):       U(-l, l), l = sqrt(6 / fan_in)            for the ReLU family.
double initializationLimit(ActivationType type, std::size_t fanIn, std::size_t fanOut) {
    switch (type) {
    case ActivationType::Sigmoid:
    case ActivationType::Tanh:
    case ActivationType::Linear:
        return std::sqrt(6.0 / static_cast<double>(fanIn + fanOut));
    case ActivationType::ReLU:
    case ActivationType::LeakyReLU:
        return std::sqrt(6.0 / static_cast<double>(fanIn));
    }
    throw std::invalid_argument("Unknown activation function type.");
}

/// A uniform double in [0, 1) with 53 random bits, built from two 32-bit engine outputs
/// (the same construction as genrand_res53 in the reference Mersenne Twister code).
///
/// std::uniform_real_distribution is deliberately not used: its algorithm is
/// implementation-defined, so libstdc++, libc++ and the MSVC STL turn the same seed into
/// different weights. std::mt19937's output sequence is fully specified by the standard and
/// every operation below is exact, so a seed produces bit-identical weights on every platform
/// that evaluates double in IEEE-754 binary64 (see ADR 0005).
// cppcheck-suppress constParameterReference ; rng() advances the engine, so it cannot be const
double uniformUnitInterval(std::mt19937& rng) {
    // mt19937 outputs are always < 2^32, whatever the width of result_type (32 bits on MSVC,
    // 64 on glibc); the widening to uint64_t is implicit because an explicit cast would be a
    // -Wuseless-cast error on GCC where the types coincide.
    const std::uint64_t high = rng() >> 5U; // top 27 bits
    const std::uint64_t low = rng() >> 6U;  // top 26 bits
    constexpr double kTwoToMinus53 = 0x1.0p-53;
    return static_cast<double>((high << 26U) | low) * kTwoToMinus53;
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

    neurons_.reserve(numberOfNeurons);
    for (std::size_t n = 0; n < numberOfNeurons; ++n) {
        std::vector<double> weights(numberOfInputsPerNeuron);
        // 2u - 1 is exact for u in [0, 1), so the result is one correctly rounded multiply and
        // cannot change under floating-point contraction (FMA) settings.
        std::ranges::generate(weights,
                              [&] { return limit * ((2.0 * uniformUnitInterval(rng)) - 1.0); });
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
    // A constructed layer always has neurons; only a moved-from layer can be empty.
    return neurons_.empty() ? 0 : neurons_.front().getNumberOfInputs();
}

const Neuron& Layer::getNeuron(std::size_t index) const {
    return neurons_.at(index);
}

Neuron& Layer::getNeuron(std::size_t index) {
    return neurons_.at(index);
}

} // namespace neuralnet
