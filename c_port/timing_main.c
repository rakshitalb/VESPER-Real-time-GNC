#include "control.h"
#include <stdint.h>

/*
 * Timing harness for the actual pd_controller().
 *
 * Three execution-path cases are tested:
 * 1. Unsaturated torque
 * 2. Positive torque saturation
 * 3. Negative torque saturation
 *
 * The inputs are volatile so the compiler cannot remove
 * the controller calls.
 */

volatile uint32_t timing_done = 0;
volatile double timing_sink[3];

static void run_controller_case(
    const double q_err[3],
    const double omega[3])
{
    double torque[3];

    pd_controller(q_err, omega, torque);

    timing_sink[0] = torque[0];
    timing_sink[1] = torque[1];
    timing_sink[2] = torque[2];
}

int main(void)
{
    /* Case 1: Unsaturated torque */
    const double q_unsat[3] =
    {
        0.0, 0.0, 0.0
    };

    const double w_unsat[3] =
    {
        0.0, 0.0, 0.0
    };

    /* Case 2: Positive torque saturation */
    const double q_positive[3] =
    {
        -1.0, -1.0, -1.0
    };

    const double w_positive[3] =
    {
        -1.0, -1.0, -1.0
    };

    /* Case 3: Negative torque saturation */
    const double q_negative[3] =
    {
        1.0, 1.0, 1.0
    };

    const double w_negative[3] =
    {
        1.0, 1.0, 1.0
    };

    /*
     * Repeat the cases so the execution trace contains
     * many actual controller executions.
     */
    for (volatile uint32_t i = 0; i < 100; i++)
    {
        run_controller_case(q_unsat, w_unsat);
        run_controller_case(q_positive, w_positive);
        run_controller_case(q_negative, w_negative);
    }

    /* Explicit completion marker for Renode inspection. */
    timing_done = 1;

    while (1)
    {
    }

    return 0;
}