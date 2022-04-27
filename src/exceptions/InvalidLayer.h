//
// Created by Hunter Harloff on 12/29/20.
//

#ifndef SPEEDNET_INVALIDLAYER_H
#define SPEEDNET_INVALIDLAYER_H

#include <exception>

namespace SpeedNet {
    class InvalidLayer : public std::exception {
    public:
        const char *what() const noexcept override;
    };
}


#endif //SPEEDNET_INVALIDLAYER_H
