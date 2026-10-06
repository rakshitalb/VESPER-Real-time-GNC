#include "FreeRTOS.h"
#include "task.h"
#include <stdint.h>

volatile uint32_t jitter_count = 0;
volatile TickType_t jitter_ticks[101];

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

static void GNC_Jitter_Task(void *arg)
{
    (void)arg;

    for (uint32_t i = 0; i < 101; i++)
    {
        jitter_ticks[i] = xTaskGetTickCount();
        jitter_count = i + 1;
        vTaskDelay(pdMS_TO_TICKS(10));
    }

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
        GNC_Jitter_Task,
        "Jitter",
        256,
        NULL,
        3,
        task_stack,
        &task_buffer
    );

    vTaskStartScheduler();

    for (;;)
    {
    }
}