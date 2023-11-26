#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
class Neuron
{
public:
	enum ActivationFunctionType
	{
		Sigmoid,
		ReLU,
		Tanh,
		LeakyReLU,
	};

	Neuron(std::vector<double> w, double b, ActivationFunctionType t);


	// calculate neuron output
	double output(const std::vector<double>& inputs);
	// calculate error value
	double calculateError(double output, double expected);
	// update parameters
	void updateParameters(const std::vector<double>& inputs, double error, double learningRate);
	void updateParameters(const std::vector<double>& inputs, double output, double expected, double learningRate);
	std::vector<double> calculateDeltas(double error);
	double derivativeOfActivationFunction(ActivationFunctionType type, double x);
	double getWeight(size_t index) const;
	double getLastOutput() const;

public:
	// activation functions

	double applyActivationFunction(ActivationFunctionType activationType, double x);

	// compresses any real number between 0 and 1.
	double sigmoid(double x);
	// ReLU sets negative inputs to zero leaves positive inputs as they are. 
	double relu(double x);
	// tanh compresses the input to a value between -1 and 1.
	double tanh(double x);
	// leaky Relu is variation of ReLU and provides small slope for negative inputs
	double leakyRelu(double x);

	double derivativeOfSigmoid(double x);
	double derivativeOfReLU(double x);
	double derivativeOfTanh(double x);
	double derivativeOfLeakyReLU(double x, double alpha = 0.01);


	// The softmax function converts the outputs of a layer of neurons into a probability distribution.
	std::vector<double> softmax(const std::vector<double>& inputs);

	std::vector<double> weights; // Weights for each input
	double bias; // Bias value
	double lastOutput; 
	ActivationFunctionType activationFunctionType;
};

