//
// Created by Poppro on 12/3/2019.
//

#include "Linear.h"

namespace SpeedNet {
    const double Linear::activate(double input) {
        return input;
    }
    
    const double Linear::derivative(double input) {
        // Derivative of linear function is constant 1
        return 1.0;
    }

    const LayerDefinition::activationType Linear::type() {
        return LayerDefinition::activationType::linear;
    }
};