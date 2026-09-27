#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuralnet {

/// Activation function applied to a neuron's weighted sum.
enum class ActivationType : std::uint8_t {
    Sigmoid,   ///< 1 / (1 + e^-x), output in (0, 1)
    ReLU,      ///< max(0, x)
    Tanh,      ///< tanh(x), output in (-1, 1)
    LeakyReLU, ///< x for x > 0, otherwise kLeakyReluSlope * x
};

/// A single artificial neuron: a = f(w . x + b).
///
/// The neuron caches its most recent output so that backpropagation can evaluate the
/// activation derivative without re-running the forward pass.
class Neuron {
public:
    /// Slope used for negative inputs by LeakyReLU.
    static constexpr double kLeakyReluSlope = 0.01;

    /// @throws std::invalid_argument if `weights` is empty.
    Neuron(std::vector<double> weights, double bias, ActivationType activationType);

    /// Computes a = f(w . x + b), caches it as the last output and returns it.
    /// @throws std::invalid_argument if `inputs.size()` differs from the number of weights.
    double output(const std::vector<double>& inputs);

    /// Applies one gradient-descent step to the weights and bias.
    /// @param inputs        The inputs used for the most recent call to output().
    /// @param error         dL/da: the loss gradient with respect to this neuron's output.
    /// @param learningRate  Step size.
    /// @throws std::invalid_argument if `inputs.size()` differs from the number of weights.
    void updateParameters(const std::vector<double>& inputs, double error, double learningRate);

    /// Evaluates the activation function `type` at `x`.
    [[nodiscard]] static double activate(ActivationType type, double x);

    /// Derivative of the activation function expressed in terms of its *output* a = f(x).
    ///
    /// Sigmoid and tanh derivatives are cheapest to compute from the output, and for the ReLU
    /// family the sign of the output equals the sign of the input, so a single convention
    /// (pass the cached output) works for every supported activation.
    [[nodiscard]] static double activationDerivative(ActivationType type, double output);

    [[nodiscard]] const std::vector<double>& getWeights() const noexcept { return weights_; }
    /// @throws std::out_of_range if `index` is invalid.
    [[nodiscard]] double getWeight(std::size_t index) const;
    /// @throws std::out_of_range if `index` is invalid.
    void setWeight(std::size_t index, double value);
    [[nodiscard]] double getBias() const noexcept { return bias_; }
    void setBias(double value) noexcept { bias_ = value; }
    [[nodiscard]] double getLastOutput() const noexcept { return lastOutput_; }
    [[nodiscard]] ActivationType getActivationType() const noexcept { return activationType_; }
    [[nodiscard]] std::size_t getNumberOfInputs() const noexcept { return weights_.size(); }

private:
    std::vector<double> weights_;
    double bias_;
    double lastOutput_ = 0.0;
    ActivationType activationType_;
};

} // namespace neuralnet
