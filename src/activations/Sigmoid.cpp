//
// Created by hunter harloff on 2019-12-01.
//

#include <cmath>
#include "Sigmoid.h"

namespace SpeedNet {
    const double Sigmoid::activate(double input) {
        return 1.0 / (1.0 + std::exp(-input));
    }
    
    const double Sigmoid::derivative(double input) {
        // Derivative of sigmoid: f(x) * (1 - f(x))
        double sigmoid = activate(input);
        return sigmoid * (1.0 - sigmoid);
    }

    const LayerDefinition::activationType Sigmoid::type() {
        return LayerDefinition::activationType::sigmoid;
    }
}
