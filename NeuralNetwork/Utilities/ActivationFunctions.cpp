// ActivationFunctions.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "ActivationFunctions.h"

float sigmoid(const float x) {
    return 1.0 / (1.0 + exp(-x));
}
