// The single most important test in the suite: it proves backpropagation computes the true
// gradient of the loss. Every weight and bias is perturbed by +/- epsilon, the loss is
// re-evaluated, and the central finite difference is compared with the analytic gradient
// that one training step applied. Any sign error, missing derivative factor or wrong index
// in the backward pass shows up here immediately.
//
// The check is only meaningful if the gradients it compares are not trivially zero. A ReLU
// unit whose pre-activation is negative passes no gradient at all, and with random weights
// most ReLU units in these small networks start out dead (before this was handled, 253 of the
// 291 ReLU parameters had an exact-zero gradient, so the check verified almost nothing). ReLU
// networks are therefore moved into their active region first, and every case asserts that
// at least half of its gradients are non-zero.
#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
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

// Sets every bias so that each unit's pre-activation on `inputs` is z = c with c drawn from
// [0.1, 1]. Every ReLU unit is then active (ReLU(z) = z) and stays well away from the kink at
// z = 0, which a finite difference with a small epsilon cannot cross.
void activateAllUnits(NeuralNetwork& network, const std::vector<double>& inputs,
                      std::mt19937& rng) {
    std::uniform_real_distribution<double> preActivation(0.1, 1.0);
    std::vector<double> current = inputs;
    for (std::size_t l = 0; l < network.getNumberOfLayers(); ++l) {
        auto& layer = network.getLayer(l);
        std::vector<double> next(layer.getNumberOfNeurons());
        for (std::size_t n = 0; n < layer.getNumberOfNeurons(); ++n) {
            auto& neuron = layer.getNeuron(n);
            const auto& weights = neuron.getWeights();
            const double weightedSum =
                std::inner_product(weights.begin(), weights.end(), current.begin(), 0.0);
            const double z = preActivation(rng);
            neuron.setBias(z - weightedSum);
            next[n] = z; // ReLU(z) = z for z > 0
        }
        current = std::move(next);
    }
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

    // The data RNG must not replay the stream that initialized the weights, or the inputs
    // would be correlated with the first layer's weights.
    std::mt19937 rng(seed + 0x9E3779B9U);
    NeuralNetwork reference(topology, activation, seed);
    const std::vector<double> inputs = randomVector(reference.getInputSize(), rng);
    const std::vector<double> targets = randomVector(reference.getOutputSize(), rng);
    if (activation == ActivationType::ReLU) {
        activateAllUnits(reference, inputs, rng);
    }

    NeuralNetwork trained = reference;
    trained.train(inputs, targets, learningRate);

    std::size_t parameters = 0;
    std::size_t nonZeroGradients = 0;
    const auto countGradient = [&](double gradient) {
        ++parameters;
        if (gradient != 0.0) {
            ++nonZeroGradients;
        }
    };

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
                countGradient(analytic);
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
            countGradient(analyticBias);
        }
    }

    // Guard against a vacuous pass: comparing zeros with zeros proves nothing.
    INFO("activation " << static_cast<int>(activation) << ": " << nonZeroGradients << " of "
                       << parameters << " gradients are non-zero");
    CHECK(nonZeroGradients * 2 >= parameters);
}
