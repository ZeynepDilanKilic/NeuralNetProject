#include "neuralnet/NeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace neuralnet {

NeuralNetwork::NeuralNetwork(std::size_t inputSize, const std::vector<LayerSpec>& layers,
                             std::uint32_t seed) {
    if (inputSize == 0) {
        throw std::invalid_argument("A network needs at least one input.");
    }
    if (layers.empty()) {
        throw std::invalid_argument("A network needs at least one layer.");
    }
    // Validate every layer before allocating anything, and say which one is wrong.
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].units == 0) {
            throw std::invalid_argument("layers[" + std::to_string(i) +
                                        "].units must be at least 1.");
        }
    }

    std::mt19937 rng(seed);
    layers_.reserve(layers.size());
    std::size_t fanIn = inputSize;
    for (const LayerSpec& spec : layers) {
        layers_.emplace_back(spec.units, fanIn, spec.activation, rng);
        fanIn = spec.units;
    }
}

namespace {

bool allFinite(const std::vector<double>& values) {
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

/// 1/2 * sum_i (outputs_i - targets_i)^2; the sizes must already agree.
double squaredErrorLoss(const std::vector<double>& outputs, const std::vector<double>& targets) {
    double loss = 0.0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const double difference = outputs[i] - targets[i];
        loss += 0.5 * difference * difference;
    }
    return loss;
}

} // namespace

void NeuralNetwork::requireLayers() const {
    if (layers_.empty()) {
        throw std::logic_error("The network has no layers (it was moved from).");
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
    requireLayers();
    if (inputs.size() != getInputSize()) {
        throw std::invalid_argument("Input size must match the network's input width.");
    }
    std::vector<double> current = inputs;
    for (Layer& layer : layers_) {
        current = layer.processInputs(current);
    }
    return current;
}

double NeuralNetwork::computeLoss(const std::vector<double>& inputs,
                                  const std::vector<double>& expectedOutputs) {
    requireLayers();
    // Validate the targets before running the forward pass, which would otherwise be wasted.
    if (expectedOutputs.size() != getOutputSize()) {
        throw std::invalid_argument("Target size must match the output size.");
    }
    return squaredErrorLoss(predict(inputs), expectedOutputs);
}

double NeuralNetwork::train(const std::vector<double>& inputs,
                            const std::vector<double>& expectedOutputs, double learningRate) {
    // Every check runs before any state changes, so a throwing call leaves the network as it was.
    requireLayers();
    if (!std::isfinite(learningRate) || learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be finite and positive.");
    }
    if (inputs.size() != getInputSize()) {
        throw std::invalid_argument("Input size must match the network's input width.");
    }
    if (expectedOutputs.size() != getOutputSize()) {
        throw std::invalid_argument("Target size must match the output size.");
    }
    // A single NaN or infinity would silently and permanently poison every weight it reaches.
    if (!allFinite(inputs)) {
        throw std::invalid_argument("Inputs must be finite.");
    }
    if (!allFinite(expectedOutputs)) {
        throw std::invalid_argument("Targets must be finite.");
    }

    // 1. Forward pass, keeping the input of every layer (activations[i] feeds layers_[i]).
    std::vector<std::vector<double>> activations;
    activations.reserve(layers_.size() + 1);
    activations.push_back(inputs);
    for (Layer& layer : layers_) {
        // cppcheck-suppress useStlAlgorithm ; each step consumes the previous layer's output
        activations.push_back(layer.processInputs(activations.back()));
    }

    // 2. Backward pass: compute dL/da for every layer before touching any weight, so that
    //    hidden-layer errors are computed with the weights used in the forward pass.
    std::vector<std::vector<double>> errors(layers_.size());
    errors.back() = calculateOutputLayerError(activations.back(), expectedOutputs);
    for (std::size_t i = layers_.size() - 1; i > 0; --i) {
        errors[i - 1] = calculateHiddenLayerError(layers_[i - 1], errors[i], layers_[i]);
    }

    // The loss of this sample under the pre-update weights comes for free from the forward pass.
    const double loss = squaredErrorLoss(activations.back(), expectedOutputs);

    // 3. Update. Each neuron multiplies dL/da by its own activation derivative exactly once.
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].updateWeights(errors[i], learningRate, activations[i]);
    }
    return loss;
}

std::vector<double>
NeuralNetwork::calculateOutputLayerError(const std::vector<double>& layerOutputs,
                                         const std::vector<double>& expectedOutputs) {
    if (layerOutputs.size() != expectedOutputs.size()) {
        throw std::invalid_argument("Target size must match the output size.");
    }
    std::vector<double> errors(layerOutputs.size());
    for (std::size_t i = 0; i < layerOutputs.size(); ++i) {
        errors[i] = layerOutputs[i] - expectedOutputs[i];
    }
    return errors;
}

std::vector<double> NeuralNetwork::calculateHiddenLayerError(
    const Layer& currentLayer, const std::vector<double>& nextLayerErrors, const Layer& nextLayer) {
    const std::size_t nextCount = nextLayer.getNumberOfNeurons();
    if (nextLayerErrors.size() != nextCount) {
        throw std::invalid_argument("Error count must match the next layer size.");
    }

    // dL/da_i (current) = sum_j delta_j * w_ji, with delta_j = dL/da_j * f'(a_j) (next).
    std::vector<double> errors(currentLayer.getNumberOfNeurons(), 0.0);
    for (std::size_t j = 0; j < nextCount; ++j) {
        const Neuron& neuron = nextLayer.getNeuron(j);
        const double nextDelta =
            nextLayerErrors[j] *
            Neuron::activationDerivative(neuron.getActivationType(), neuron.getLastOutput());
        for (std::size_t i = 0; i < errors.size(); ++i) {
            errors[i] += nextDelta * neuron.getWeight(i);
        }
    }
    return errors;
}

// A constructed network always has layers; only a moved-from network can be empty, and its
// accessors must stay well-defined (they are noexcept and must not dereference front()/back()).
std::size_t NeuralNetwork::getInputSize() const noexcept {
    return layers_.empty() ? 0 : layers_.front().getNumberOfInputs();
}

std::size_t NeuralNetwork::getOutputSize() const noexcept {
    return layers_.empty() ? 0 : layers_.back().getNumberOfNeurons();
}

const Layer& NeuralNetwork::getLayer(std::size_t index) const {
    return layers_.at(index);
}

Layer& NeuralNetwork::getLayer(std::size_t index) {
    return layers_.at(index);
}

} // namespace neuralnet
