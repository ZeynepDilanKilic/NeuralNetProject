#include "Layer.h"
Layer::Layer(int numberOfNeurons, int numberOfInputsPerNeuron, Neuron::ActivationFunctionType activationType)
{
	for (size_t i = 0; i < numberOfNeurons; i++)
	{
		std::vector<double> weights(numberOfInputsPerNeuron, 0.1); // set starting weights to a default value
		double bias = 0.1; // set starting bias value
		neurons.emplace_back(weights, bias, activationType);
	}
}

std::vector<double> Layer::processInputs(const std::vector<double>& inputs)
{
	std::vector<double> outputs;
	for (auto& neuron : neurons)
	{
		outputs.push_back(neuron.output(inputs));
	}
	return outputs;
}

void Layer::updateWeights(const std::vector<double>& errors, double learningRate, const std::vector<double>& inputs) {
	for (size_t i = 0; i < neurons.size(); ++i) {
		neurons[i].updateParameters(inputs, errors[i], learningRate);
	}
}

int Layer::getNumberOfNeurons() const
{
	return neurons.size();
}

const Neuron& Layer::getNeuron(size_t index) const
{
	return neurons.at(index);
}

