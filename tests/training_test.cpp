#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

using Catch::Approx;
using neuralnet::ActivationType;
using neuralnet::LayerSpec;
using neuralnet::NeuralNetwork;

TEST_CASE("Neurons in the same layer start with different weights", "[init]") {
    // Regression test: constant initialization (all weights 0.1) made every neuron in a
    // layer identical forever, because identical neurons receive identical gradients.
    NeuralNetwork network(3, {{4, ActivationType::Sigmoid}, {2, ActivationType::Sigmoid}});
    const auto& layer = network.getLayer(0);
    for (std::size_t a = 0; a < layer.getNumberOfNeurons(); ++a) {
        for (std::size_t b = a + 1; b < layer.getNumberOfNeurons(); ++b) {
            CHECK(layer.getNeuron(a).getWeights() != layer.getNeuron(b).getWeights());
        }
    }
}

TEST_CASE("Initial weights follow the Glorot/He uniform distribution", "[init]") {
    // A wide first layer (fan_in = 300, fan_out = 200) gives enough samples to check the
    // distribution itself, not just a bound: U(-l, l) has max |w| close to l and variance l^2/3.
    // A wrong fan (e.g. fan_out where fan_in belongs) changes l and fails these checks.
    constexpr std::size_t fanIn = 300;
    constexpr std::size_t fanOut = 200;
    const auto [activation, limit] = GENERATE(table<ActivationType, double>({
        {ActivationType::Sigmoid, std::sqrt(6.0 / static_cast<double>(fanIn + fanOut))},
        {ActivationType::Tanh, std::sqrt(6.0 / static_cast<double>(fanIn + fanOut))},
        {ActivationType::Linear, std::sqrt(6.0 / static_cast<double>(fanIn + fanOut))},
        {ActivationType::ReLU, std::sqrt(6.0 / static_cast<double>(fanIn))},
        {ActivationType::LeakyReLU, std::sqrt(6.0 / static_cast<double>(fanIn))},
    }));
    INFO("activation " << static_cast<int>(activation));

    NeuralNetwork network(fanIn, {{fanOut, activation}, {1, activation}});
    const auto& layer = network.getLayer(0);
    double maxAbs = 0.0;
    double sum = 0.0;
    double sumOfSquares = 0.0;
    std::size_t count = 0;
    for (std::size_t n = 0; n < layer.getNumberOfNeurons(); ++n) {
        CHECK(layer.getNeuron(n).getBias() == 0.0);
        for (double w : layer.getNeuron(n).getWeights()) {
            maxAbs = std::max(maxAbs, std::abs(w));
            sum += w;
            sumOfSquares += w * w;
            ++count;
        }
    }
    const double mean = sum / static_cast<double>(count);
    const double variance = (sumOfSquares / static_cast<double>(count)) - (mean * mean);

    CHECK(maxAbs <= limit);
    CHECK(maxAbs > 0.99 * limit);
    CHECK(std::abs(mean) < 0.01 * limit);
    CHECK(variance == Approx(limit * limit / 3.0).epsilon(0.02));
}

TEST_CASE("Initialization is identical on every platform", "[init]") {
    // Golden values: the seed alone determines the weights, bit for bit, on every compiler
    // and standard library (std::uniform_real_distribution is not used; see ADR 0005). If this
    // test fails, saved experiments and published results no longer reproduce. The values are
    // hexadecimal floating-point literals so they are exact.
    NeuralNetwork network(2, {{2, ActivationType::Tanh}, {1, ActivationType::ReLU}}, 42);
    const auto& hidden = network.getLayer(0);
    const auto& output = network.getLayer(1);
    CHECK(hidden.getNeuron(0).getWeight(0) == -0x1.3ab02da762f3ap-2);
    CHECK(hidden.getNeuron(0).getWeight(1) == 0x1.1aa10f30d33cbp+0);
    CHECK(hidden.getNeuron(1).getWeight(0) == 0x1.22f3dd0e598a5p-1);
    CHECK(hidden.getNeuron(1).getWeight(1) == 0x1.eeecfb6e293p-3);
    CHECK(output.getNeuron(0).getWeight(0) == -0x1.310bce224aa37p+0);
    CHECK(output.getNeuron(0).getWeight(1) == -0x1.311147f221ccdp+0);
}

TEST_CASE("Initialization is deterministic for a given seed", "[init]") {
    const std::vector<LayerSpec> layers{{3, ActivationType::Tanh}, {1, ActivationType::Tanh}};
    NeuralNetwork a(2, layers, 123);
    NeuralNetwork b(2, layers, 123);
    NeuralNetwork c(2, layers, 124);
    const std::vector<double> inputs{0.25, -0.75};

    CHECK(a.predict(inputs) == b.predict(inputs));
    CHECK(a.predict(inputs) != c.predict(inputs));
}

TEST_CASE("One gradient step on a single neuron reduces the loss", "[training]") {
    const auto activation =
        GENERATE(ActivationType::Sigmoid, ActivationType::Tanh, ActivationType::ReLU,
                 ActivationType::LeakyReLU, ActivationType::Linear);
    NeuralNetwork network(1, {{1, activation}});
    // Fix the parameters instead of relying on what the seed draws: with a random weight the
    // ReLU neuron is dead (zero gradient) for about half of all seeds.
    auto& neuron = network.getLayer(0).getNeuron(0);
    neuron.setWeight(0, 0.3);
    neuron.setBias(0.0);
    const std::vector<double> inputs{1.0};
    const std::vector<double> targets{0.5};

    const double lossBefore = network.computeLoss(inputs, targets);
    network.train(inputs, targets, 0.1);
    const double lossAfter = network.computeLoss(inputs, targets);

    INFO("activation " << static_cast<int>(activation));
    CHECK(std::isfinite(lossAfter));
    CHECK(lossAfter < lossBefore);
}

TEST_CASE("A dead ReLU neuron receives no gradient", "[training]") {
    // Documents the known ReLU failure mode: with z = w * x + b <= 0 the output and its
    // derivative are both 0, so training cannot move the neuron out of the dead region.
    NeuralNetwork network(1, {{1, ActivationType::ReLU}});
    auto& neuron = network.getLayer(0).getNeuron(0);
    neuron.setWeight(0, -0.3);
    neuron.setBias(0.0);
    const std::vector<double> inputs{1.0};
    const std::vector<double> targets{0.5};

    const double loss = network.train(inputs, targets, 0.1);

    CHECK(loss == Approx(0.125)); // 1/2 * (0 - 0.5)^2
    CHECK(network.getLayer(0).getNeuron(0).getWeight(0) == -0.3);
    CHECK(network.getLayer(0).getNeuron(0).getBias() == 0.0);
}

TEST_CASE("One gradient step applies exactly w -= learningRate * dL/dw", "[training]") {
    // Single sigmoid neuron, x = 1, w = 0.3, b = 0, target 0.5:
    // dL/dz = (a - y) * a * (1 - a), dL/dw = dL/dz * x, dL/db = dL/dz.
    NeuralNetwork network(1, {{1, ActivationType::Sigmoid}});
    auto& neuron = network.getLayer(0).getNeuron(0);
    neuron.setWeight(0, 0.3);
    neuron.setBias(0.0);
    constexpr double learningRate = 0.1;

    const double a = 1.0 / (1.0 + std::exp(-0.3));
    const double delta = (a - 0.5) * a * (1.0 - a);
    network.train({1.0}, {0.5}, learningRate);

    CHECK(network.getLayer(0).getNeuron(0).getWeight(0) ==
          Approx(0.3 - (learningRate * delta)).epsilon(1e-12));
    CHECK(network.getLayer(0).getNeuron(0).getBias() ==
          Approx(-learningRate * delta).epsilon(1e-12));
}

TEST_CASE("train() returns the loss before the update", "[training]") {
    const auto activation =
        GENERATE(ActivationType::Sigmoid, ActivationType::Tanh, ActivationType::ReLU,
                 ActivationType::LeakyReLU, ActivationType::Linear);
    NeuralNetwork network(2, {{3, activation}, {2, activation}}, 7);
    const std::vector<double> inputs{0.4, -0.6};
    const std::vector<double> targets{0.2, 0.9};

    const double expected = network.computeLoss(inputs, targets);
    const double returned = network.train(inputs, targets, 0.1);

    INFO("activation " << static_cast<int>(activation));
    CHECK(returned == expected);
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

    NeuralNetwork network(2, {{4, ActivationType::Tanh}, {1, ActivationType::Sigmoid}}, 42);
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

TEST_CASE("A ReLU network with a linear output fits targets outside [0, 1]", "[training]") {
    // Regression that the old single-activation API could not express: a ReLU output cannot
    // produce negative values, and sigmoid/tanh outputs cannot exceed 1 in magnitude.
    // Fit y = 3x - 2 on x in [-1, 1], i.e. targets in [-5, 1].
    NeuralNetwork network(1, {{8, ActivationType::ReLU}, {1, ActivationType::Linear}}, 3);
    const std::array<double, 5> xs{-1.0, -0.5, 0.0, 0.5, 1.0};
    for (int epoch = 0; epoch < 2000; ++epoch) {
        for (double x : xs) {
            network.train({x}, {(3.0 * x) - 2.0}, 0.01);
        }
    }
    for (double x : xs) {
        INFO("x = " << x);
        CHECK(network.predict({x}).at(0) == Approx((3.0 * x) - 2.0).margin(0.05));
    }
}
