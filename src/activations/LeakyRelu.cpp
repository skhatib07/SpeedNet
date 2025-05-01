//
// Created on 2025-05-01.
//

#include <cmath>
#include "LeakyRelu.h"

namespace SpeedNet {
    const double LeakyRelu::activate(double input) {
        return input >= 0 ? input : alpha * input;
    }
    
    const double LeakyRelu::derivative(double input) {
        return input >= 0 ? 1.0 : alpha;
    }

    const LayerDefinition::activationType LeakyRelu::type() {
        return LayerDefinition::activationType::leakyRelu;
    }
}