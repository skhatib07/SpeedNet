//
// Created by Samer Khatib on 4/26/22.
//

#ifndef SPEEDNET_INVALIDNETWORKSIZE_H
#define SPEEDNET_INVALIDNETWORKSIZE_H

#include <exception>

namespace SpeedNet {
    class InvalidNetworkSize : public std::exception {
    public:
        const char *what() const noexcept override;
    };
}


#endif //SPEEDNET_INVALIDNETWORKSIZE_H
