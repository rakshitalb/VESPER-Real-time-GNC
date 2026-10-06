#include "FreeRTOS.h"
#include "task.h"
#include "control.h"
#include <stdint.h>

#define MEASUREMENTS 100

volatile uint32_t measurement_count = 0;
volatile uint32_t period_min_ticks = 0xFFFFFFFF;
volatile uint32_t period_max_ticks = 0;
volatile uint32_t execution_min_ticks = 0xFFFFFFFF;
volatile uint32_t execution_max_ticks = 0;
volatile uint32_t execution_spread_ticks = 0;
volatile uint32_t final_jitter_ticks = 0;

volatile TickType_t activation_ticks[MEASUREMENTS + 1];

static StaticTask_t idle_tcb;
static StackType_t idle_stack[configMINIMAL_STACK_SIZE];

void vApplicationGetIdleTaskMemory(
    StaticTask_t **ppxIdleTaskTCBBuffer,
    StackType_t **ppxIdleTaskStackBuffer,
    configSTACK_DEPTH_TYPE *pulIdleTaskStackSize)
{
    *ppxIdleTaskTCBBuffer = &idle_tcb;
    *ppxIdleTaskStackBuffer = idle_stack;
    *pulIdleTaskStackSize = configMINIMAL_STACK_SIZE;
}

void vApplicationStackOverflowHook(
    TaskHandle_t xTask,
    char *pcTaskName)
{
    (void)xTask;
    (void)pcTaskName;

    taskDISABLE_INTERRUPTS();

    for (;;)
    {
    }
}

static void GNC_Measurement_Task(void *arg)
{
    (void)arg;

    const double q_err[3] = {0.01, -0.02, 0.03};
    const double omega[3] = {0.10, -0.05, 0.02};
    double torque[3];

    TickType_t previous_tick = xTaskGetTickCount();

    activation_ticks[0] = previous_tick;

    for (uint32_t i = 0; i < MEASUREMENTS; i++)
    {
        TickType_t start_tick;
        TickType_t end_tick;
        TickType_t current_tick;
        uint32_t period;
        uint32_t execution;

        start_tick = xTaskGetTickCount();

        current_tick = start_tick;
        activation_ticks[i + 1] = current_tick;

        if (i > 0)
        {
            period = (uint32_t)(current_tick - previous_tick);

            if (period < period_min_ticks)
                period_min_ticks = period;

            if (period > period_max_ticks)
                period_max_ticks = period;
        }

        previous_tick = current_tick;

        pd_controller(q_err, omega, torque);

        end_tick = xTaskGetTickCount();

        execution = (uint32_t)(end_tick - start_tick);

        if (execution < execution_min_ticks)
            execution_min_ticks = execution;

        if (execution > execution_max_ticks)
            execution_max_ticks = execution;

        measurement_count = i + 1;

        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (period_max_ticks >= period_min_ticks)
        final_jitter_ticks = period_max_ticks - period_min_ticks;

    if (execution_max_ticks >= execution_min_ticks)
        execution_spread_ticks =
            execution_max_ticks - execution_min_ticks;

    for (;;)
    {
        vTaskDelay(portMAX_DELAY);
    }
}

int main(void)
{
    static StaticTask_t task_buffer;
    static StackType_t task_stack[256];

    xTaskCreateStatic(
        GNC_Measurement_Task,
        "GNC_MEAS",
        256,
        NULL,
        4,
        task_stack,
        &task_buffer
    );

    vTaskStartScheduler();

    for (;;)
    {
    }
}