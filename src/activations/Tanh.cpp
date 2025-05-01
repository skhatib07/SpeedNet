//
// Created by Poppro on 12/3/2019.
//

#include <cmath>
#include "Tanh.h"

namespace SpeedNet {
    const double Tanh::activate(double input) {
        return (2 / (1 + exp(-2 * input)) - 1);
    }
    
    const double Tanh::derivative(double input) {
        // Derivative of tanh: 1 - tanh^2(x)
        double tanhValue = activate(input);
        return 1.0 - tanhValue * tanhValue;
    }

    const LayerDefinition::activationType Tanh::type() {
        return LayerDefinition::activationType::tanh;
    }
}
