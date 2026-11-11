#include "neuralnet/NeuralNetwork.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
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
    SECTION("non-finite inputs and targets") {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        constexpr double inf = std::numeric_limits<double>::infinity();
        CHECK_THROWS_AS(network.train({nan, 0.0}, targets, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train({0.0, inf}, targets, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train({-inf, 0.0}, targets, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, {nan}, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(network.train(inputs, {inf}, 0.1), std::invalid_argument);
    }
    SECTION("valid call") {
        CHECK_NOTHROW(network.train(inputs, targets, 0.1));
    }
}

TEST_CASE("A rejected training call leaves every parameter unchanged", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Tanh);
    const auto snapshot = [&network] {
        std::vector<double> parameters;
        for (std::size_t l = 0; l < network.getNumberOfLayers(); ++l) {
            const auto& layer = network.getLayer(l);
            for (std::size_t n = 0; n < layer.getNumberOfNeurons(); ++n) {
                const auto& weights = layer.getNeuron(n).getWeights();
                parameters.insert(parameters.end(), weights.begin(), weights.end());
                parameters.push_back(layer.getNeuron(n).getBias());
            }
        }
        return parameters;
    };
    const std::vector<double> before = snapshot();
    const std::vector<double> inputs{1.0, 0.0};
    const std::vector<double> targets{1.0};

    CHECK_THROWS(network.train({1.0}, targets, 0.1));
    CHECK_THROWS(network.train(inputs, {1.0, 0.0}, 0.1));
    CHECK_THROWS(network.train(inputs, targets, -1.0));
    CHECK_THROWS(network.train({std::numeric_limits<double>::quiet_NaN(), 0.0}, targets, 0.1));
    CHECK_THROWS(network.train(inputs, {std::numeric_limits<double>::infinity()}, 0.1));

    CHECK(snapshot() == before); // bit-for-bit
}

TEST_CASE("computeLoss rejects a target vector of the wrong width", "[validation]") {
    NeuralNetwork network({2, 3, 1}, ActivationType::Sigmoid);
    CHECK_THROWS_AS(network.computeLoss({1.0, 0.0}, {1.0, 2.0}), std::invalid_argument);
    CHECK_THROWS_AS(network.computeLoss({1.0, 0.0}, {}), std::invalid_argument);
    CHECK_THROWS_AS(network.computeLoss({1.0}, {1.0}), std::invalid_argument);
}

TEST_CASE("A moved-from network is safe to query and refuses to run", "[validation]") {
    NeuralNetwork source({2, 3, 1}, ActivationType::Sigmoid);
    const NeuralNetwork destination = std::move(source);
    CHECK(destination.getInputSize() == std::size_t{2});

    // Querying a moved-from object is deliberate here: the accessors are noexcept and used to
    // dereference front()/back() of an empty vector (undefined behaviour).
    // NOLINTBEGIN(bugprone-use-after-move,hicpp-invalid-access-moved)
    // cppcheck-suppress accessMoved
    CHECK(source.getNumberOfLayers() == std::size_t{0});
    // cppcheck-suppress accessMoved
    CHECK(source.getInputSize() == std::size_t{0});
    // cppcheck-suppress accessMoved
    CHECK(source.getOutputSize() == std::size_t{0});
    // cppcheck-suppress accessMoved
    CHECK_THROWS_AS(source.predict({}), std::logic_error);
    // cppcheck-suppress accessMoved
    CHECK_THROWS_AS(source.train({}, {}, 0.1), std::logic_error);
    // cppcheck-suppress accessMoved
    CHECK_THROWS_AS(source.computeLoss({}, {}), std::logic_error);
    // NOLINTEND(bugprone-use-after-move,hicpp-invalid-access-moved)

    // A moved-from object can be assigned to again and is then fully usable.
    source = destination;
    CHECK(source.getInputSize() == std::size_t{2});
    CHECK_NOTHROW(source.predict({1.0, 0.0}));
}

TEST_CASE("A moved-from layer reports zero inputs instead of invoking UB", "[validation]") {
    std::mt19937 rng(0);
    Layer source(3, 2, ActivationType::Tanh, rng);
    const Layer destination = std::move(source);
    CHECK(destination.getNumberOfInputs() == std::size_t{2});

    // NOLINTBEGIN(bugprone-use-after-move,hicpp-invalid-access-moved)
    // cppcheck-suppress accessMoved
    CHECK(source.getNumberOfNeurons() == std::size_t{0});
    // cppcheck-suppress accessMoved
    CHECK(source.getNumberOfInputs() == std::size_t{0});
    // NOLINTEND(bugprone-use-after-move,hicpp-invalid-access-moved)
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
