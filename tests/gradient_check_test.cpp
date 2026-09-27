// The single most important test in the suite: it proves backpropagation computes the true
// gradient of the loss. Every weight and bias is perturbed by +/- epsilon, the loss is
// re-evaluated, and the central finite difference is compared with the analytic gradient
// that one training step applied. Any sign error, missing derivative factor or wrong index
// in the backward pass shows up here immediately.
#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using Catch::Approx;
using neuralnet::ActivationType;
using neuralnet::NeuralNetwork;

namespace {

std::vector<double> randomVector(std::size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> values(size);
    std::ranges::generate(values, [&] { return distribution(rng); });
    return values;
}

// Central finite difference of the loss w.r.t. one parameter, using a fresh copy of the
// network for each evaluation so the reference network is never modified.
template <typename SetParameter>
double numericGradient(const NeuralNetwork& reference, const std::vector<double>& inputs,
                       const std::vector<double>& targets, double original, double epsilon,
                       SetParameter setParameter) {
    NeuralNetwork plus = reference;
    setParameter(plus, original + epsilon);
    NeuralNetwork minus = reference;
    setParameter(minus, original - epsilon);
    return (plus.computeLoss(inputs, targets) - minus.computeLoss(inputs, targets)) /
           (2.0 * epsilon);
}

} // namespace

TEST_CASE("Backpropagation gradients match finite differences", "[gradient]") {
    const auto activation = GENERATE(ActivationType::Sigmoid, ActivationType::Tanh,
                                     ActivationType::ReLU, ActivationType::LeakyReLU);
    const auto topology =
        GENERATE(values<std::vector<std::size_t>>({{1}, {3, 4, 2}, {2, 5, 5, 1}}));
    const auto seed = GENERATE(as<std::uint32_t>{}, 1U, 7U, 2024U);

    constexpr double epsilon = 1e-5;
    constexpr double tolerance = 1e-6;
    // With learningRate = 1 a single step is w_after = w_before - dL/dw, so the analytic
    // gradient is recovered exactly as (w_before - w_after).
    constexpr double learningRate = 1.0;

    std::mt19937 rng(seed);
    const NeuralNetwork reference(topology, activation, seed);
    const std::vector<double> inputs = randomVector(reference.getInputSize(), rng);
    const std::vector<double> targets = randomVector(reference.getOutputSize(), rng);

    NeuralNetwork trained = reference;
    trained.train(inputs, targets, learningRate);

    for (std::size_t l = 0; l < reference.getNumberOfLayers(); ++l) {
        const auto& layer = reference.getLayer(l);
        for (std::size_t n = 0; n < layer.getNumberOfNeurons(); ++n) {
            const auto& neuron = layer.getNeuron(n);
            const auto& trainedNeuron = trained.getLayer(l).getNeuron(n);

            for (std::size_t w = 0; w < neuron.getNumberOfInputs(); ++w) {
                const double analytic = neuron.getWeight(w) - trainedNeuron.getWeight(w);
                const double numeric =
                    numericGradient(reference, inputs, targets, neuron.getWeight(w), epsilon,
                                    [l, n, w](NeuralNetwork& net, double value) {
                                        net.getLayer(l).getNeuron(n).setWeight(w, value);
                                    });
                INFO("activation " << static_cast<int>(activation) << ", layer " << l << ", neuron "
                                   << n << ", weight " << w);
                CHECK(analytic == Approx(numeric).margin(tolerance));
            }

            const double analyticBias = neuron.getBias() - trainedNeuron.getBias();
            const double numericBias =
                numericGradient(reference, inputs, targets, neuron.getBias(), epsilon,
                                [l, n](NeuralNetwork& net, double value) {
                                    net.getLayer(l).getNeuron(n).setBias(value);
                                });
            INFO("activation " << static_cast<int>(activation) << ", layer " << l << ", neuron "
                               << n << ", bias");
            CHECK(analyticBias == Approx(numericBias).margin(tolerance));
        }
    }
}
