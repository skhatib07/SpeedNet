//
// Created by Poppro on 12/4/2019.
//

#ifndef SPEEDNET_INVALIDINPUTSIZE_H
#define SPEEDNET_INVALIDINPUTSIZE_H

#include <exception>

namespace SpeedNet {
    class InvalidInputSize : public std::exception {
    public:
        const char *what() const noexcept override;
    };
}

#endif //SPEEDNET_INVALIDINPUTSIZE_H
