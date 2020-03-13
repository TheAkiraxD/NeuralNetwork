// ActivationFunctions.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "ActivationFunctions.h"

#define M_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define ERF_COEF      (1.0/M_2_SQRTPI)

float sigmoid(const float x) {
    return erf(ERF_COEF * x);
}
