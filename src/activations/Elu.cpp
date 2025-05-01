//
// Created on 2025-05-01.
//

#include <cmath>
#include "Elu.h"

namespace SpeedNet {
    const double Elu::activate(double input) {
        return input >= 0 ? input : alpha * (std::exp(input) - 1);
    }
    
    const double Elu::derivative(double input) {
        return input >= 0 ? 1.0 : alpha * std::exp(input);
    }

    const LayerDefinition::activationType Elu::type() {
        return LayerDefinition::activationType::elu;
    }
}