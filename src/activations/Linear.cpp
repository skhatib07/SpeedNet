//
// Created by Poppro on 12/3/2019.
//

#include "Linear.h"

namespace SpeedNet {
    const double Linear::activate(double input) {
        return input;
    }

    const LayerDefinition::activationType Linear::type() {
        return LayerDefinition::activationType::linear;
    }
};