#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <cmath>
#include <vector>

using Catch::Approx;
using neuralnet::ActivationType;
using neuralnet::NeuralNetwork;

TEST_CASE("Neurons in the same layer start with different weights", "[init]") {
    // Regression test: constant initialization (all weights 0.1) made every neuron in a
    // layer identical forever, because identical neurons receive identical gradients.
    NeuralNetwork network({3, 4, 2}, ActivationType::Sigmoid);
    const auto& layer = network.getLayer(1);
    for (std::size_t a = 0; a < layer.getNumberOfNeurons(); ++a) {
        for (std::size_t b = a + 1; b < layer.getNumberOfNeurons(); ++b) {
            CHECK(layer.getNeuron(a).getWeights() != layer.getNeuron(b).getWeights());
        }
    }
}

TEST_CASE("Initial weights are bounded by the Glorot/He limit", "[init]") {
    // Layer 1 of {3, 4, 2}: fan_in = 3, fan_out = 4.
    SECTION("Glorot for sigmoid: sqrt(6 / (fan_in + fan_out))") {
        NeuralNetwork network({3, 4, 2}, ActivationType::Sigmoid);
        const double limit = std::sqrt(6.0 / (3.0 + 4.0));
        for (std::size_t n = 0; n < 4; ++n) {
            for (double w : network.getLayer(1).getNeuron(n).getWeights()) {
                CHECK(std::abs(w) <= limit);
            }
            CHECK(network.getLayer(1).getNeuron(n).getBias() == 0.0);
        }
    }
    SECTION("He for ReLU: sqrt(6 / fan_in)") {
        NeuralNetwork network({3, 4, 2}, ActivationType::ReLU);
        const double limit = std::sqrt(6.0 / 3.0);
        for (std::size_t n = 0; n < 4; ++n) {
            for (double w : network.getLayer(1).getNeuron(n).getWeights()) {
                CHECK(std::abs(w) <= limit);
            }
        }
    }
}

TEST_CASE("Initialization is deterministic for a given seed", "[init]") {
    NeuralNetwork a({2, 3, 1}, ActivationType::Tanh, 123);
    NeuralNetwork b({2, 3, 1}, ActivationType::Tanh, 123);
    NeuralNetwork c({2, 3, 1}, ActivationType::Tanh, 124);
    const std::vector<double> inputs{0.25, -0.75};

    CHECK(a.predict(inputs) == b.predict(inputs));
    CHECK(a.predict(inputs) != c.predict(inputs));
}

TEST_CASE("One gradient step on a single neuron reduces the loss", "[training]") {
    const auto activation = GENERATE(ActivationType::Sigmoid, ActivationType::Tanh,
                                     ActivationType::ReLU, ActivationType::LeakyReLU);
    NeuralNetwork network({1}, activation);
    const std::vector<double> inputs{1.0};
    const std::vector<double> targets{0.5};

    const double lossBefore = network.computeLoss(inputs, targets);
    network.train(inputs, targets, 0.1);
    const double lossAfter = network.computeLoss(inputs, targets);

    INFO("activation " << static_cast<int>(activation));
    CHECK(std::isfinite(lossAfter));
    CHECK(lossAfter < lossBefore);
}

TEST_CASE("A small network learns XOR", "[training]") {
    struct Sample {
        std::vector<double> inputs;
        std::vector<double> target;
    };
    const std::array<Sample, 4> dataset{{
        {.inputs = {0.0, 0.0}, .target = {0.0}},
        {.inputs = {0.0, 1.0}, .target = {1.0}},
        {.inputs = {1.0, 0.0}, .target = {1.0}},
        {.inputs = {1.0, 1.0}, .target = {0.0}},
    }};

    NeuralNetwork network({2, 4, 1}, ActivationType::Tanh, 42);
    constexpr int epochs = 3000;
    constexpr double learningRate = 0.5;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const Sample& sample : dataset) {
            network.train(sample.inputs, sample.target, learningRate);
        }
    }

    for (const Sample& sample : dataset) {
        const double prediction = network.predict(sample.inputs).at(0);
        INFO("inputs " << sample.inputs[0] << ", " << sample.inputs[1]);
        CHECK(std::abs(prediction - sample.target[0]) < 0.1);
    }
}
