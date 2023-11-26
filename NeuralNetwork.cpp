#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes, Neuron::ActivationFunctionType activationType) {
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        int numberOfInputsPerNeuron = (i == 0) ? layerSizes[i] : layerSizes[i - 1];
        layers.emplace_back(layerSizes[i], numberOfInputsPerNeuron, activationType);
    }
}

void NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs, double learningRate) {
    // Ýleri besleme (Feedforward) - Katmanlarý sýrayla iþle
    std::vector<double> currentInputs = inputs;
    //currentInputs = feedForward(currentInputs);
    backPropagate(currentInputs,expectedOutputs,learningRate);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs)
{
    std::vector<double> currentInputs = inputs;
    for (auto& layer : layers) {
        currentInputs = layer.processInputs(currentInputs);
    }
    return currentInputs; // Son katmanýn çýktýlarý
}

std::vector<double> NeuralNetwork::calculateOutputLayerError(const std::vector<double>& layerOutputs, const std::vector<double>& expectedOutputs) {
    std::vector<double> errors(layerOutputs.size());
    for (size_t i = 0; i < layerOutputs.size(); ++i) {
        errors[i] = layerOutputs[i] - expectedOutputs[i];
    }
    return errors;
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputValues) {
    std::vector<double> currentInputs = inputValues;
    for (auto& layer : layers) {
        currentInputs = layer.processInputs(currentInputs);
    }
    return currentInputs;
}
std::vector<double> NeuralNetwork::calculateHiddenLayerError(const Layer& currentLayer, const std::vector<double>& nextLayerErrors, const Layer& nextLayer) {
    std::vector<double> hiddenLayerErrors(currentLayer.getNumberOfNeurons());

    for (size_t i = 0; i < currentLayer.getNumberOfNeurons(); ++i) {
        double error = 0.0;
        for (size_t j = 0; j < nextLayer.getNumberOfNeurons(); ++j) {
            error += nextLayerErrors[j] * nextLayer.getNeuron(j).getWeight(i);
        }
        //currentLayer.getNeuron(i)
       
        hiddenLayerErrors[i] = error * derivativeOfSigmoid(currentLayer.getNeuron(i).getLastOutput());
    }

    return hiddenLayerErrors;
}

void NeuralNetwork::backPropagate(const std::vector<double>& inputs, const std::vector<double>& expectedOutputs, double learningRate) {
    // save output layer and do forward feed
    std::vector<double> currentInputs = inputs;
    std::vector<std::vector<double>> layerOutputs;
    for (auto& layer : layers) {
        currentInputs = layer.processInputs(currentInputs);
        layerOutputs.push_back(currentInputs);
    }

    // Geri yayýlým
    std::vector<double> errors;
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
            // calculate for output layer
            errors.resize(layers[i].getNumberOfNeurons());
            for (size_t j = 0; j < errors.size(); ++j) {
                errors[j] = expectedOutputs[j] - layerOutputs.back()[j];
            }
        }
        else {
            //error calculation for hidden layer
            auto& nextLayer = layers[i + 1];
            std::vector<double> newErrors(layers[i].getNumberOfNeurons(), 0.0);
            for (size_t j = 0; j < newErrors.size(); ++j) {
                double error = 0.0;
                for (size_t k = 0; k < nextLayer.getNumberOfNeurons(); ++k) {
                    error += errors[k] * nextLayer.getNeuron(k).getWeight(j);
                }
                newErrors[j] = error * derivativeOfSigmoid(layerOutputs[i][j]);
            }
            errors = newErrors;
        }

        // update weights
        std::vector<double> inputsToLayer = (i == 0) ? inputs : layerOutputs[i - 1];
        layers[i].updateWeights(errors, learningRate, inputsToLayer);
    }
}
//bunlarý neurondan çaðýrmak gerekiyor.
double NeuralNetwork::derivativeOfSigmoid(double output) {
    return output * (1 - output);
}

double NeuralNetwork::derivativeOfReLU(double input) {
    return input > 0 ? 1.0 : 0.0;
}