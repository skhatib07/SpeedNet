//
// Created by hunter harloff on 2019-12-01.
//

#include "InvalidInputSize.h"

namespace SpeedNet {
    const char* InvalidInputSize::what() const noexcept {
        return "Invalid size supplied to Perceptron.";
    }
}
