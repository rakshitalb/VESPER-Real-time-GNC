#ifndef WEEK9_INTERFACE_H
#define WEEK9_INTERFACE_H

typedef struct
{
    unsigned long timestamp;
    unsigned long status;
    double torque_x;
    double torque_y;
    double torque_z;
} GNC_Telemetry;

extern volatile GNC_Telemetry gnc_telemetry;

#endif