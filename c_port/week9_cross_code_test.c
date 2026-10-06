#include "control.h"
#include <stdint.h>
volatile int32_t test_torque_micro[3];
volatile uint32_t test_done=0;
int main(void){
    const double q[3]={0.5,-0.25,0.1};
    const double w[3]={0.2,-0.1,0.05};
    double t[3];
    pd_controller(q,w,t);
    test_torque_micro[0]=(int32_t)(t[0]*1000000.0);
    test_torque_micro[1]=(int32_t)(t[1]*1000000.0);
    test_torque_micro[2]=(int32_t)(t[2]*1000000.0);
    test_done=1;
    while(1){}
}
