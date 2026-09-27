#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using neuralnet::ActivationType;
using neuralnet::Layer;
using neuralnet::NeuralNetwork;
using neuralnet::Neuron;

TEST_CASE("Construction rejects degenerate shapes", "[validation]") {
    std::mt19937 rng(0);
    CHECK_THROWS_AS(NeuralNetwork({}, ActivationType::Sigmoid), std::invalid_argument);
    CHECK_THROWS_AS(NeuralNetwork({2, 0, 1}, ActivationType::Sigmoid), std::invalid_argument);
    CHECK_THROWS_AS(Layer(0, 3, ActivationType::ReLU, rng), std::invalid_argument);
    CHECK_THROWS_AS(Layer(3, 0, ActivationType::ReLU, rng), std::invalid_argument);
    CHECK_THROWS_AS(Neuron({}, 0.0, ActivationType::ReLU), std::invalid_argument);
}

TEST_CASE("Neuron and Layer reject mismatched vector sizes on their own", "[validation]") {
    // These checks are normally shielded by NeuralNetwork's validation; they still have to hold
    // when the lower-level classes are used directly.
    std::mt19937 rng(0);
    Neuron neuron({0.5, -0.5}, 0.0, ActivationType::ReLU);
    CHECK_THROWS_AS(neuron.output({1.0}), std::invalid_argument);
    CHECK_THROWS_AS(neuron.updateParameters({1.0, 2.0, 3.0}, 0.1, 0.1), std::invalid_argument);

    Layer layer(2, 3, ActivationType::ReLU, rng);
    const std::vector<double> inputs{1.0, 2.0, 3.0};
    CHECK_NOTHROW(layer.processInputs(inputs));
    CHECK_THROWS_AS(layer.updateWeights({0.1}, 0.1, inputs), std::invalid_argument);
    CHECK_NOTHROW(layer.updateWeights({0.1, 0.2}, 0.1, inputs));
}

TEST_CASE("Forward pass rejects an input vector of the wrong width", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Sigmoid);
    CHECK_THROWS_AS(network.predict({1.0}), std::invalid_argument);
    CHECK_THROWS_AS(network.predict({1.0, 2.0, 3.0}), std::invalid_argument);
    CHECK_NOTHROW(network.predict({1.0, 2.0}));
}

TEST_CASE("Training validates every argument", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Sigmoid);
    const std::vector<double> inputs{1.0, 0.0};
    const std::vector<double> targets{1.0};

    SECTION("input width") {
        CHECK_THROWS_AS(network.train({1.0}, targets, 0.1), std::invalid_argument);
    }
    SECTION("target width") {
        CHECK_THROWS_AS(network.train(inputs, {1.0, 0.0}, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, {}, 0.1), std::invalid_argument);
    }
    SECTION("learning rate") {
        CHECK_THROWS_AS(network.train(inputs, targets, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, targets, -0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, targets, std::numeric_limits<double>::quiet_NaN()),
                        std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, targets, std::numeric_limits<double>::infinity()),
                        std::invalid_argument);
    }
    SECTION("valid call") {
        CHECK_NOTHROW(network.train(inputs, targets, 0.1));
    }
}

TEST_CASE("Accessors are bounds-checked", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Sigmoid);
    CHECK_THROWS_AS(network.getLayer(3), std::out_of_range);
    CHECK_THROWS_AS(network.getLayer(1).getNeuron(3), std::out_of_range);
    CHECK_THROWS_AS(network.getLayer(1).getNeuron(0).getWeight(2), std::out_of_range);
    CHECK_THROWS_AS(network.getLayer(1).getNeuron(0).setWeight(2, 0.0), std::out_of_range);
    CHECK_NOTHROW(network.getLayer(2).getNeuron(0).getWeight(2));
}

TEST_CASE("Network shape accessors report the configured topology", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Sigmoid);
    CHECK(network.getNumberOfLayers() == std::size_t{3});
    CHECK(network.getInputSize() == std::size_t{2});
    CHECK(network.getOutputSize() == std::size_t{1});
    CHECK(network.getLayer(1).getNumberOfNeurons() == std::size_t{3});
    CHECK(network.getLayer(1).getNumberOfInputs() == std::size_t{2});
    CHECK(network.getLayer(2).getNumberOfInputs() == std::size_t{3});
    CHECK(network.getLayer(2).getActivationType() == ActivationType::Sigmoid);
}
