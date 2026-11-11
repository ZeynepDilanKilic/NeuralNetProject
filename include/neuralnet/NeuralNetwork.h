#pragma once

#include "neuralnet/Layer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuralnet {

/// One trainable, fully connected layer of a NeuralNetwork, e.g. `{4, ActivationType::Tanh}`.
///
/// The constructor (rather than a plain aggregate) makes the activation mandatory: an
/// aggregate would silently value-initialize an omitted activation to Sigmoid.
struct LayerSpec {
    constexpr LayerSpec(std::size_t unitCount, ActivationType layerActivation) noexcept
        : units(unitCount), activation(layerActivation) {}

    // A plain value type: public fields are the point (`spec.units`), the constructor only
    // exists to make the activation mandatory, and there is no invariant to protect.
    // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
    std::size_t units;         ///< Number of neurons (outputs) in the layer; must be > 0.
    ActivationType activation; ///< Activation applied by every neuron in the layer.
    // NOLINTEND(misc-non-private-member-variables-in-classes)
};

/// A feed-forward network trained with per-sample stochastic gradient descent on the
/// squared-error loss L = 1/2 * sum_i (a_i - y_i)^2.
class NeuralNetwork {
public:
    static constexpr std::uint32_t kDefaultSeed = 42;

    /// Builds a network that takes `inputSize` inputs and applies `layers` in order; the last
    /// entry is the output layer. The input itself is not a layer and has no weights.
    ///
    /// @code
    /// // 2 inputs -> 4 tanh hidden units -> 1 sigmoid output
    /// NeuralNetwork network(2, {{4, ActivationType::Tanh}, {1, ActivationType::Sigmoid}});
    /// @endcode
    ///
    /// @param inputSize  Width of the input vector; must be > 0.
    /// @param layers     The trainable layers, from the first hidden layer to the output.
    /// @param seed       Seed for weight initialization. Initialization is portable: the same
    ///                   seed produces bit-identical weights with every compiler and standard
    ///                   library on platforms that evaluate double in IEEE-754 binary64
    ///                   (x86-64, ARM64, MSVC; not 32-bit x87 builds), see ADR 0005.
    /// @throws std::invalid_argument if `inputSize` is 0, `layers` is empty or any layer has
    ///         zero units. All arguments are checked before any memory is allocated.
    NeuralNetwork(std::size_t inputSize, const std::vector<LayerSpec>& layers,
                  std::uint32_t seed = kDefaultSeed);

    /// Runs a forward pass and returns the output layer's activations.
    ///
    /// Non-finite inputs are not rejected here. Depending on the activation the output may be
    /// NaN, infinite or a saturated finite value (Sigmoid/Tanh squash +/-inf), so validate
    /// inputs yourself if that matters.
    /// @throws std::invalid_argument if `inputs.size()` differs from getInputSize().
    /// @throws std::logic_error if the network was moved from.
    [[nodiscard]] std::vector<double> predict(const std::vector<double>& inputs);

    /// Performs one gradient-descent step on a single (inputs, expectedOutputs) pair.
    ///
    /// Exception guarantee: every argument is validated before any weight or bias changes, and
    /// the update phase itself cannot throw, so if this throws no weight or bias has changed.
    /// (Cached per-neuron outputs, see Neuron::getLastOutput(), may have been overwritten if the
    /// forward pass already ran.)
    /// @returns The loss of this sample *before* the update, i.e. what computeLoss() would have
    ///          returned had it been called first. Using it avoids a second forward pass.
    /// @throws std::invalid_argument on size mismatches, non-finite inputs or targets, or a
    ///         non-positive/non-finite learning rate.
    /// @throws std::logic_error if the network was moved from.
    double train(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs,
                 double learningRate);

    /// Returns 1/2 * sum_i (predict(inputs)_i - expectedOutputs_i)^2.
    /// @throws std::invalid_argument on size mismatches.
    /// @throws std::logic_error if the network was moved from.
    [[nodiscard]] double computeLoss(const std::vector<double>& inputs,
                                     const std::vector<double>& expectedOutputs);

    /// Shape accessors. A moved-from network has no layers and reports 0 for all of them.
    [[nodiscard]] std::size_t getNumberOfLayers() const noexcept { return layers_.size(); }
    [[nodiscard]] std::size_t getInputSize() const noexcept;
    [[nodiscard]] std::size_t getOutputSize() const noexcept;
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] const Layer& getLayer(std::size_t index) const;
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] Layer& getLayer(std::size_t index);

private:
    /// @throws std::logic_error if `layers_` is empty, which only happens after a move.
    void requireLayers() const;

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
