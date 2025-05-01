//
// Created on 2025-05-01.
//

#include <cmath>
#include "Selu.h"

namespace SpeedNet {
    const double Selu::activate(double input) {
        // SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
        return scale * (input > 0 ? input : alpha * (std::exp(input) - 1));
    }
    
    const double Selu::derivative(double input) {
        // Derivative of SELU
        if (input > 0) {
            return scale;
        } else {
            return scale * alpha * std::exp(input);
        }
    }

    const LayerDefinition::activationType Selu::type() {
        return LayerDefinition::activationType::selu;
    }
}