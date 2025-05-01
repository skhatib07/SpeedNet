//
// Created by Poppro on 12/3/2019.
//

#include <algorithm>
#include "Relu.h"

namespace SpeedNet {
    const double Relu::activate(double input) {
        return std::max(0.0, input);
    }
    
    const double Relu::derivative(double input) {
        // Derivative of ReLU: 1 if x > 0, 0 otherwise
        return input > 0 ? 1.0 : 0.0;
    }

    const LayerDefinition::activationType Relu::type() {
        return LayerDefinition::activationType::relu;
    }
}
