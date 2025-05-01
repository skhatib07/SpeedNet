//
// Created by Poppro on 12/3/2019.
//

#include "Step.h"

namespace SpeedNet {
    const double Step::activate(double input) {
        return (input >= 0) ? 1 : 0;
    }
    
    const double Step::derivative(double input) {
        // Step function is not differentiable at 0, and has derivative 0 elsewhere
        // For backpropagation purposes, we can use a small constant or 0
        return 0.0;
    }

    const LayerDefinition::activationType Step::type() {
        return LayerDefinition::activationType::step;
    }
}
