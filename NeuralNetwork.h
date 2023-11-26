#pragma once
#include "Layer.h"

class NeuralNetwork
{
private:
	std::vector<Layer> layers;
public:
	NeuralNetwork(const std::vector<int>& layerSizes, Neuron::ActivationFunctionType activationType);
	void train(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs, double learningRate);
	std::vector<double> calculateOutputLayerError(const std::vector<double>& layerOutputs, const std::vector<double>& expectedOutputs);
	std::vector<double> calculateHiddenLayerError(const Layer& currentLayer, const std::vector<double>& nextLayerErrors, const Layer& nextLayer);
	std::vector<double> feedForward(const std::vector<double>& inputValues);
	void backPropagate(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs, double learningRate);
	std::vector<double> predict(const std::vector<double>& inputs);
	double derivativeOfSigmoid(double output);
	double derivativeOfReLU(double output);
};

