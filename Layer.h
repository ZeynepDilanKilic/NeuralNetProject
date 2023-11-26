#pragma once
#include "Neuron.h"

//An artificial neural network usually consists of one or more layers.Each layer contains a set of neurons and their activations.
//The Layer class may include functions for computing these neurons and the output of the layer.


class Layer
{
public:
	Layer(int numberOfNeurons, int numberOfInputsPerNeuron, Neuron::ActivationFunctionType activationType);
	std::vector<double> processInputs(const std::vector<double>& inputs);
	void updateWeights(const std::vector<double>& errors, double learningRate, const std::vector<double>& inputs);
	int getNumberOfNeurons() const;
	const Neuron& getNeuron(size_t index) const;
	


	/*void calculateGradients(const std::vector<double>& errors);
	std::vector<double> getOutputGradients();
	std::vector<double> calculateDerivatives();
	void initializeWeights();
	void applyRegularization(double regularizationRate);
	void applyDropout(double dropoutRate);
	double calculateError(const std::vector<double>& expectedOutputs);*/
private:
	std::vector<Neuron> neurons;
};

