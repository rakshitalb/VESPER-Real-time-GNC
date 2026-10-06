#include "control.h"
#include <stdint.h>

/*
 * Results exposed in RAM for Renode inspection.
 */
volatile uint32_t functional_pass;
volatile uint32_t benchmark_done;
volatile uint32_t benchmark_result;

/*
 * Benchmark markers.
 */
__attribute__((noinline))
void benchmark_start(void)
{
}

__attribute__((noinline))
void benchmark_end(void)
{
}

/*
 * Functional equivalence test.
 *
 * The PD controller itself remains unchanged.
 * This verifies the C implementation against
 * the expected reference result.
 */
static uint32_t check_functional_equivalence(void)
{
    double q_err[3] =
    {
        0.5,
       -0.25,
        0.1
    };

    double omega[3] =
    {
        0.2,
       -0.1,
        0.05
    };

    double torque[3];

    const double tolerance = 1.0e-9;

    const double expected_x = -0.0100;
    const double expected_y =  0.0070;
    const double expected_z = -0.0033;

    pd_controller(q_err, omega, torque);

    if ((torque[0] - expected_x > tolerance) ||
        (expected_x - torque[0] > tolerance))
        return 0;

    if ((torque[1] - expected_y > tolerance) ||
        (expected_y - torque[1] > tolerance))
        return 0;

    if ((torque[2] - expected_z > tolerance) ||
        (expected_z - torque[2] > tolerance))
        return 0;

    return 1;
}

/*
 * Integer workload used only for deterministic
 * Renode execution benchmarking.
 */
static uint32_t benchmark_workload(void)
{
    uint32_t value = 0x12345678;

    for (volatile uint32_t i = 0; i < 100000; i++)
    {
        value = value * 1664525u + 1013904223u;
        value ^= (value >> 13);
    }

    return value;
}

int main(void)
{
    /*
     * First verify functional equivalence.
     */
    functional_pass = check_functional_equivalence();

    /*
     * Controlled benchmark.
     */
    benchmark_start();

    benchmark_result = benchmark_workload();

    benchmark_end();

    /*
     * Explicit RAM completion marker.
     */
    benchmark_done = 1;

    /*
     * Keep CPU alive after benchmark.
     */
    while (1)
    {
    }

    return 0;
}