#include "neuralnet/Neuron.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>

using Catch::Approx;
using neuralnet::ActivationType;
using neuralnet::Neuron;

TEST_CASE("Activation functions produce the expected values", "[activation]") {
    SECTION("Sigmoid") {
        CHECK(Neuron::activate(ActivationType::Sigmoid, 0.0) == Approx(0.5));
        CHECK(Neuron::activate(ActivationType::Sigmoid, 100.0) == Approx(1.0));
        CHECK(Neuron::activate(ActivationType::Sigmoid, -100.0) == Approx(0.0).margin(1e-12));
        CHECK(std::isfinite(Neuron::activate(ActivationType::Sigmoid, -1000.0)));
    }

    SECTION("ReLU") {
        CHECK(Neuron::activate(ActivationType::ReLU, -2.0) == 0.0);
        CHECK(Neuron::activate(ActivationType::ReLU, 0.0) == 0.0);
        CHECK(Neuron::activate(ActivationType::ReLU, 3.5) == 3.5);
    }

    SECTION("Tanh matches std::tanh and does not overflow") {
        const double x = GENERATE(-1000.0, -2.0, 0.0, 2.0, 1000.0);
        const double value = Neuron::activate(ActivationType::Tanh, x);
        CHECK(std::isfinite(value)); // the hand-rolled (e^x - e^-x)/(e^x + e^-x) gave NaN here
        CHECK(value == Approx(std::tanh(x)));
    }

    SECTION("LeakyReLU") {
        CHECK(Neuron::activate(ActivationType::LeakyReLU, -2.0) ==
              Approx(-2.0 * Neuron::kLeakyReluSlope));
        CHECK(Neuron::activate(ActivationType::LeakyReLU, 3.5) == 3.5);
    }
}

TEST_CASE("Activation derivatives agree with a central finite difference", "[activation]") {
    const auto type = GENERATE(ActivationType::Sigmoid, ActivationType::ReLU, ActivationType::Tanh,
                               ActivationType::LeakyReLU);
    const double x = GENERATE(-3.0, -0.5, 0.7, 2.5);
    constexpr double epsilon = 1e-6;

    const double numeric =
        (Neuron::activate(type, x + epsilon) - Neuron::activate(type, x - epsilon)) /
        (2.0 * epsilon);
    // The derivative API takes the activation *output*, not the pre-activation.
    const double analytic = Neuron::activationDerivative(type, Neuron::activate(type, x));

    INFO("activation " << static_cast<int>(type) << " at x = " << x);
    CHECK(analytic == Approx(numeric).margin(1e-6));
}
