#pragma once

#include "neuralnet/Layer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuralnet {

/// A feed-forward network trained with per-sample stochastic gradient descent on the
/// squared-error loss L = 1/2 * sum_i (a_i - y_i)^2.
class NeuralNetwork {
public:
    static constexpr std::uint32_t kDefaultSeed = 42;

    /// @param layerSizes      layerSizes[i] is the number of neurons in layer i.
    ///                        NOTE: layerSizes[0] doubles as the network's input width, so the
    ///                        first layer is a trainable layer with layerSizes[0] inputs and
    ///                        layerSizes[0] neurons. Making the input width an explicit
    ///                        parameter is on the roadmap.
    /// @param activationType  Activation used by every layer.
    /// @param seed            Seed for weight initialization; the same seed and standard
    ///                        library always produce the same network.
    /// @throws std::invalid_argument if `layerSizes` is empty or contains a zero.
    NeuralNetwork(const std::vector<std::size_t>& layerSizes, ActivationType activationType,
                  std::uint32_t seed = kDefaultSeed);

    /// Runs a forward pass and returns the output layer's activations.
    /// @throws std::invalid_argument if `inputs.size()` differs from getInputSize().
    [[nodiscard]] std::vector<double> predict(const std::vector<double>& inputs);

    /// Performs one gradient-descent step on a single (inputs, expectedOutputs) pair.
    /// @throws std::invalid_argument on size mismatches or a non-positive/non-finite rate.
    void train(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs,
               double learningRate);

    /// Returns 1/2 * sum_i (predict(inputs)_i - expectedOutputs_i)^2.
    /// @throws std::invalid_argument on size mismatches.
    [[nodiscard]] double computeLoss(const std::vector<double>& inputs,
                                     const std::vector<double>& expectedOutputs);

    [[nodiscard]] std::size_t getNumberOfLayers() const noexcept { return layers_.size(); }
    [[nodiscard]] std::size_t getInputSize() const noexcept;
    [[nodiscard]] std::size_t getOutputSize() const noexcept;
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] const Layer& getLayer(std::size_t index) const;
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] Layer& getLayer(std::size_t index);

private:
    /// dL/da for the output layer under the squared-error loss: a - y.
    [[nodiscard]] static std::vector<double>
    calculateOutputLayerError(const std::vector<double>& layerOutputs,
                              const std::vector<double>& expectedOutputs);

    /// dL/da for `currentLayer`, given dL/da of `nextLayer` (which consumes its outputs).
    [[nodiscard]] static std::vector<double>
    calculateHiddenLayerError(const Layer& currentLayer, const std::vector<double>& nextLayerErrors,
                              const Layer& nextLayer);

    std::vector<Layer> layers_;
};

} // namespace neuralnet
