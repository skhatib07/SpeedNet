//
// Created on 2025-05-01.
//

#include <cmath>
#include "Softplus.h"

namespace SpeedNet {
    const double Softplus::activate(double input) {
        // Softplus: f(x) = ln(1 + e^x)
        return std::log(1.0 + std::exp(input));
    }
    
    const double Softplus::derivative(double input) {
        // Derivative of Softplus: f'(x) = 1 / (1 + e^(-x))
        // This is the sigmoid function
        return 1.0 / (1.0 + std::exp(-input));
    }

    const LayerDefinition::activationType Softplus::type() {
        return LayerDefinition::activationType::softplus;
    }
}