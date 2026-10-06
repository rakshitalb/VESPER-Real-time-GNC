#include "control.h"
#include <stdio.h>
#include <math.h>

#define KP 0.008
#define KD 0.05
#define TAU_MAX 0.010
double limit_torque(double torque)
{
    if (torque > TAU_MAX)
        return TAU_MAX;

    if (torque < -TAU_MAX)
        return -TAU_MAX;

    return torque;
}
void pd_controller(const double q_err[3],
                   const double omega[3],
                   double torque[3])
{
    for (int i = 0; i < 3; i++)
    {
        torque[i] = -KP * q_err[i] - KD * omega[i];

        torque[i] = limit_torque(torque[i]);
    }
}
