#pragma once

#include "neuralnet/Neuron.h"

#include <cstddef>
#include <random>
#include <vector>

namespace neuralnet {

/// A fully connected layer: every neuron receives the same input vector.
class Layer {
public:
    /// Creates `numberOfNeurons` neurons with `numberOfInputsPerNeuron` weights each.
    ///
    /// Weights are drawn from `rng` with Glorot (Xavier) uniform initialization for
    /// Sigmoid/Tanh and He uniform initialization for ReLU/LeakyReLU. Biases start at zero.
    /// Random initialization is what breaks the symmetry between neurons; if every neuron
    /// started with identical weights they would receive identical gradients and never
    /// diverge, collapsing the layer to a single effective neuron.
    ///
    /// @throws std::invalid_argument if either size is zero.
    Layer(std::size_t numberOfNeurons, std::size_t numberOfInputsPerNeuron,
          ActivationType activationType, std::mt19937& rng);

    /// Runs every neuron on `inputs` and returns their activations in neuron order.
    [[nodiscard]] std::vector<double> processInputs(const std::vector<double>& inputs);

    /// Applies one gradient step to every neuron.
    /// @param errors        errors[i] = dL/da_i, the loss gradient w.r.t. neuron i's output.
    /// @param learningRate  Step size.
    /// @param inputs        The vector last passed to processInputs().
    /// @throws std::invalid_argument if `errors.size()` differs from the number of neurons.
    void updateWeights(const std::vector<double>& errors, double learningRate,
                       const std::vector<double>& inputs);

    [[nodiscard]] std::size_t getNumberOfNeurons() const noexcept { return neurons_.size(); }
    [[nodiscard]] std::size_t getNumberOfInputs() const noexcept;
    [[nodiscard]] ActivationType getActivationType() const noexcept { return activationType_; }
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] const Neuron& getNeuron(std::size_t index) const;
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] Neuron& getNeuron(std::size_t index);

private:
    std::vector<Neuron> neurons_;
    ActivationType activationType_;
};

} // namespace neuralnet
