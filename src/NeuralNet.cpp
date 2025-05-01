#include "NeuralNet.h"

namespace SpeedNet {
    // Constructor

    NeuralNet::NeuralNet(int inputSize){
        this->inputSize = inputSize;
    }

    NeuralNet::NeuralNet(int inputSize, const std::vector<LayerDefinition> &layers) {
        ActivationFactory* activationFactory = ActivationFactory::getInstance();
        this->inputSize = inputSize;
        this->layers.reserve(layers.size());
        int prevSize = inputSize;
        for (LayerDefinition layer : layers) {
            Activation* activation = activationFactory->getActivation(layer.activation);
            this->layers.emplace_back(layer.size, prevSize, activation);
            prevSize = layer.size;
        }
    }

    NeuralNet::NeuralNet(std::istream &is) {
        this->inputSize = 0;
        is >> *this;
    }

    // Predictor

    std::vector<double> NeuralNet::predict(const std::vector<double> &input) {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        if (input.size() != this->inputSize) throw InvalidInputSize();
        std::vector<double> layerData(input);
        for (Layer &l : this->layers) {
            layerData = l.calculate(layerData);
        }
        return layerData;
    }

    // Insertor
    void NeuralNet::add(const LayerDefinition &newLayer){

        ActivationFactory* activationFactory = ActivationFactory::getInstance();

        this->layers.reserve(this->layers.size() + 1);
        int prevSize;
        if(this->layers.empty()){prevSize = this->inputSize;}
        else{prevSize = this->layers.back().size();}

        Activation* activation = activationFactory->getActivation(newLayer.activation);
        this->layers.emplace_back(newLayer.size, prevSize, activation);
    }

    // Mutators

    void NeuralNet::mutate_uniform(const double lower, const double upper, int layer) {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        if (layer == -1) {
            for (Layer &l : this->layers) {
                l.mutate_uniform(lower, upper);
            }
        } else {
            if (layer < 0 || layer >= this->layers.size())
                throw InvalidLayer();
            this->layers[layer].mutate_uniform(lower, upper);
        }
    }

    void NeuralNet::mutate_gaussian(const double mean, const double std, int layer) {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        if (layer == -1) {
            for (Layer &l : this->layers) {
                l.mutate_gaussian(mean, std);
            }
        } else {
            if (layer < 0 || layer >= this->layers.size())
                throw InvalidLayer();
            this->layers[layer].mutate_gaussian(mean, std);
        }
    }

    // Getters

    const int NeuralNet::getInputSize() {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        return this->inputSize;
    }

    const int NeuralNet::getOutputSize() {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        return this->layers.back().size();
    }

    // Serialize

    // Write to file
    std::ostream &operator<<(std::ostream &os, const NeuralNet& nn) {
        os << nn.inputSize << " ";
        os << nn.layers.size() << " ";
        for (const Layer& l : nn.layers)
            os << l << " ";
        return os;
    }

    // Read from file
    std::istream& operator>>(std::istream& is, NeuralNet& nn) {
        is >> nn.inputSize;
        int layerSize;
        is >> layerSize;
        nn.layers.resize(layerSize, Layer(0, 0, nullptr));
        for (Layer& l : nn.layers)
            is >> l;
        return is;
    }
    
    // Backpropagation implementation
    
    double NeuralNet::train(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
        if (this->layers.size() < 3 || this->inputSize == 0) throw InvalidNetworkSize();
        if (input.size() != this->inputSize) throw InvalidInputSize();
        if (target.size() != this->layers.back().size()) throw InvalidInputSize();
        
        // Forward pass - store outputs of each layer
        std::vector<std::vector<double>> layerOutputs;
        layerOutputs.reserve(this->layers.size() + 1); // +1 for input layer
        
        // Store input as first "layer output"
        layerOutputs.push_back(input);
        
        // Forward propagation
        std::vector<double> currentInput = input;
        for (Layer &layer : this->layers) {
            std::vector<double> output = layer.calculate(currentInput);
            layerOutputs.push_back(output);
            currentInput = output;
        }
        
        // Backpropagation
        backpropagate(target, layerOutputs, learningRate);
        
        // Calculate and return error
        return calculateError(layerOutputs.back(), target);
    }
    
    double NeuralNet::trainBatch(const std::vector<std::vector<double>>& inputs,
                                const std::vector<std::vector<double>>& targets,
                                double learningRate,
                                int batchSize) {
        if (inputs.size() != targets.size()) {
            throw InvalidInputSize();
        }
        
        double totalError = 0.0;
        int actualBatchSize = std::min(static_cast<int>(inputs.size()), batchSize);
        
        for (int i = 0; i < actualBatchSize; i++) {
            totalError += train(inputs[i], targets[i], learningRate);
        }
        
        return totalError / actualBatchSize;
    }
    
    void NeuralNet::backpropagate(const std::vector<double>& target, std::vector<std::vector<double>>& layerOutputs, double learningRate) {
        // Calculate deltas for each layer, starting from the output layer
        std::vector<std::vector<double>> layerDeltas;
        layerDeltas.resize(this->layers.size());
        
        // Calculate output layer deltas
        int outputLayerIndex = this->layers.size() - 1;
        layerDeltas[outputLayerIndex] = this->layers[outputLayerIndex].calculateOutputDeltas(
            target, layerOutputs[outputLayerIndex + 1]);
        
        // Calculate hidden layer deltas (working backwards)
        for (int i = outputLayerIndex - 1; i >= 0; i--) {
            // Collect weights from the next layer
            std::vector<std::vector<double>> nextLayerWeights;
            for (int j = 0; j < this->layers[i + 1].size(); j++) {
                nextLayerWeights.push_back(this->layers[i + 1].getWeights(j));
            }
            
            layerDeltas[i] = this->layers[i].calculateDeltas(
                layerDeltas[i + 1], nextLayerWeights, layerOutputs[i + 1]);
        }
        
        // Update weights for all layers
        for (size_t i = 0; i < this->layers.size(); i++) {
            this->layers[i].updateWeights(layerDeltas[i], layerOutputs[i], learningRate);
        }
    }
    
    double NeuralNet::calculateError(const std::vector<double>& predicted, const std::vector<double>& target) {
        // Calculate mean squared error
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); i++) {
            double error = target[i] - predicted[i];
            sum += error * error;
        }
        return sum / predicted.size();
    }
}
