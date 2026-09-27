#include "neuralnet/NeuralNetwork.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace neuralnet {

NeuralNetwork::NeuralNetwork(const std::vector<std::size_t>& layerSizes,
                             ActivationType activationType, std::uint32_t seed) {
    if (layerSizes.empty()) {
        throw std::invalid_argument("A network needs at least one layer.");
    }

    std::mt19937 rng(seed);
    layers_.reserve(layerSizes.size());
    for (std::size_t i = 0; i < layerSizes.size(); ++i) {
        const std::size_t numberOfInputsPerNeuron = (i == 0) ? layerSizes[0] : layerSizes[i - 1];
        layers_.emplace_back(layerSizes[i], numberOfInputsPerNeuron, activationType, rng);
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
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
    const std::vector<double> outputs = predict(inputs);
    if (outputs.size() != expectedOutputs.size()) {
        throw std::invalid_argument("Target size must match the output size.");
    }
    double loss = 0.0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const double difference = outputs[i] - expectedOutputs[i];
        loss += 0.5 * difference * difference;
    }
    return loss;
}

void NeuralNetwork::train(const std::vector<double>& inputs,
                          const std::vector<double>& expectedOutputs, double learningRate) {
    if (!std::isfinite(learningRate) || learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be finite and positive.");
    }
    if (inputs.size() != getInputSize()) {
        throw std::invalid_argument("Input size must match the network's input width.");
    }
    if (expectedOutputs.size() != getOutputSize()) {
        throw std::invalid_argument("Target size must match the output size.");
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

    // 3. Update. Each neuron multiplies dL/da by its own activation derivative exactly once.
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].updateWeights(errors[i], learningRate, activations[i]);
    }
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

std::size_t NeuralNetwork::getInputSize() const noexcept {
    return layers_.front().getNumberOfInputs();
}

std::size_t NeuralNetwork::getOutputSize() const noexcept {
    return layers_.back().getNumberOfNeurons();
}

const Layer& NeuralNetwork::getLayer(std::size_t index) const {
    return layers_.at(index);
}

Layer& NeuralNetwork::getLayer(std::size_t index) {
    return layers_.at(index);
}

} // namespace neuralnet
