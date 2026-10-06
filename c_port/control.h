#ifndef CONTROL_H
#define CONTROL_H

void pd_controller(const double q_err[3],
                   const double omega[3],
                   double torque[3]);

#endif