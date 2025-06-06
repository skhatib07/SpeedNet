//
// Created by hunter harloff on 2019-12-01.
//

#include "Perceptron.h"

#include "random/RandomGenerator.h"
#include "exceptions/InvalidInputSize.h"

namespace SpeedNet {
    // Constructor

    Perceptron::Perceptron(int inputSize) {
        this->inputSize = inputSize;
        this->weights.resize(inputSize + 1);
    };

    // Mutators

    void Perceptron::calculate(const std::vector<double> &input, Activation *activation) {
        if (input.size() != this->inputSize) {
            throw InvalidInputSize();
        }
        double weightedSum = 0;
        for (int i = 0; i < this->inputSize; i++) {
            weightedSum += input[i] * this->weights[i];
        }
        weightedSum += 1 * this->weights.back(); // bias term
        this->cachedWeightedSum = weightedSum; // Store for backpropagation
        this->cachedValue = activation->activate(weightedSum);
    }

    void Perceptron::mutate_uniform(const double lower, const double upper) {
        RandomGenerator *randomGenerator = RandomGenerator::getInstance();
        for (double &weight : this->weights) {
            weight += randomGenerator->generate_uniform(lower, upper);
        }
    }

    void Perceptron::mutate_gaussian(const double mean, const double std) {
        RandomGenerator *randomGenerator = RandomGenerator::getInstance();
        for (double &weight : this->weights) {
            weight += randomGenerator->generate_gaussian(mean, std);
        }
    }

    // Getters

    double Perceptron::getValue() const {
        return this->cachedValue;
    }
    
    double Perceptron::getWeightedSum() const {
        return this->cachedWeightedSum;
    }
    
    void Perceptron::updateWeights(double delta, const std::vector<double>& inputs, double learningRate) {
        // Update weights based on delta and inputs
        for (int i = 0; i < this->inputSize; i++) {
            this->weights[i] += learningRate * delta * inputs[i];
        }
        // Update bias weight (using 1 as input)
        this->weights.back() += learningRate * delta * 1.0;
    }
    
    const std::vector<double>& Perceptron::getWeights() const {
        return this->weights;
    }

    // Serialize

    std::ostream &operator<<(std::ostream &os, const Perceptron &p) {
        os << p.inputSize << " ";
        os << p.weights.size() << " ";
        for (double w : p.weights)
            os << w << " ";
        return os;
    }

    std::istream& operator>>(std::istream& is, Perceptron &p) {
        is >> p.inputSize;
        int weightSize;
        is >> weightSize;
        p.weights.resize(weightSize);
        for (double& w : p.weights)
            is >> w;
        return is;
    }
}