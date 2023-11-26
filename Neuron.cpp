#include "Neuron.h"
#include <iostream>
Neuron::Neuron(std::vector<double> w, double b, ActivationFunctionType t) : weights(w), bias(b), activationFunctionType(t), lastOutput(0.0)
{
};

double Neuron::calculateError(double output, double expected)
{
	return 0.5 * std::pow(output - expected, 2);
}

double Neuron::output(const std::vector<double>& intputs)
{
	double total = 0.0;
	for (size_t i = 0; i < weights.size(); ++i)
	{
		total += weights[i] * intputs[i];
	}
	total += bias;
	lastOutput = applyActivationFunction(ActivationFunctionType::Sigmoid, total);
	return lastOutput;
}

void Neuron::updateParameters(const std::vector<double>& inputs, double output, double expected, double learningRate)
{
	double derivative = (output - expected) * output * (1 - output); // derivative of eror function
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights[i] -= learningRate * derivative * inputs[i];
	}

	bias -= learningRate * derivative;
}

void Neuron::updateParameters(const std::vector<double>& inputs, double error, double learningRate) {
	double derivative = lastOutput * (1 - lastOutput); // Sigmoid için
	for (size_t i = 0; i < weights.size(); ++i) {
		weights[i] -= learningRate * derivative * error * inputs[i];
	}
	bias -= learningRate * derivative * error;
}


std::vector<double> Neuron::calculateDeltas(double error)
{
	std::vector<double> deltas(weights.size());
	for (size_t i = 0; i < weights.size(); ++i) {
		// Aktivasyon fonksiyonunun türevi ile hata çarpýlýr.
		// Örneðin, sigmoid aktivasyon fonksiyonu için:
		double derivative = lastOutput * (1 - lastOutput); // sigmoid'in türevi
		deltas[i] = error * derivative;
	}
	return deltas;
}


double Neuron::applyActivationFunction(ActivationFunctionType activationType, double x)
{
	switch (activationType)
	{
	case ActivationFunctionType::Sigmoid:
		return sigmoid(x);
	case ActivationFunctionType::ReLU:
		return relu(x);
	case ActivationFunctionType::Tanh:
		return tanh(x);
	case ActivationFunctionType::LeakyReLU:
		return leakyRelu(x);
	default:
		throw std::invalid_argument("Invalid activation function type");
	}
}


double Neuron::derivativeOfActivationFunction(ActivationFunctionType type, double x) 
{
	switch (type)
	{
	case ActivationFunctionType::Sigmoid:
		return derivativeOfSigmoid(x);
	case ActivationFunctionType::ReLU:
		return derivativeOfReLU(x);
	case ActivationFunctionType::Tanh:
		return derivativeOfTanh(x);
	case ActivationFunctionType::LeakyReLU:
		return derivativeOfLeakyReLU(x);
	default:
		throw std::invalid_argument("Invalid activation function type");
	}
}

std::vector<double> Neuron::softmax(const std::vector<double>& inputs)
{

	std::vector<double> exponentials(inputs.size());
	std::transform(inputs.begin(), inputs.end(), exponentials.begin(), [](double input) {
		return std::exp(input);
		});

	double sumOfExponentials = std::accumulate(exponentials.begin(), exponentials.end(), 0.0);

	std::vector<double> output(inputs.size());
	std::transform(exponentials.begin(), exponentials.end(), output.begin(),[sumOfExponentials](double expValue)
		{
			return expValue / sumOfExponentials;
		});
	return output;

}

double Neuron::sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

double Neuron::relu(double x)
{
	return (x > 0) ? x : 0;
}

double Neuron::tanh(double x)
{
	return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}
double Neuron::leakyRelu(double x) {
	return (x > 0) ? x : 0.01 * x;
}
double Neuron::getWeight(size_t index) const
{
	return weights.at(index);
}

double Neuron::getLastOutput() const
{
	return lastOutput;
}

//bunlarý neurondan çaðýrmak gerekiyor.
double Neuron::derivativeOfSigmoid(double x) {
	return x * (1 - x);
}

double Neuron::derivativeOfReLU(double x) {
	return x > 0 ? 1.0 : 0.0;
}

double Neuron::derivativeOfTanh(double x) {
	return 1.0 - std::pow(x, 2);
}

double Neuron::derivativeOfLeakyReLU(double x, double alpha) {
	return x > 0 ? 1.0 : alpha;
}