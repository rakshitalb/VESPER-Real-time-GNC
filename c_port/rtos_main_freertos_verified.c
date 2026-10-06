#include "FreeRTOS.h"
#include "task.h"
volatile unsigned long gnc_count = 0;
volatile unsigned long sensing_count = 0;
volatile unsigned long communications_count = 0;
volatile unsigned long monitoring_count = 0;
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
static void GNC_Task(void *pvParameters)
{
    (void)pvParameters;

    for (;;)
{
    gnc_count++;
    /* GNC/control placeholder */
    vTaskDelay(pdMS_TO_TICKS(10));
}
}

static void Sensing_Task(void *pvParameters)
{
    (void)pvParameters;

    for (;;)
{
    sensing_count++;
    /* Sensing/estimation placeholder */
    vTaskDelay(pdMS_TO_TICKS(20));
}
}

static void Communications_Task(void *pvParameters)
{
    (void)pvParameters;

   for (;;)
{
    communications_count++;
    /* Communications placeholder */
    vTaskDelay(pdMS_TO_TICKS(50));
}
}

static void Monitoring_Task(void *pvParameters)
{
    (void)pvParameters;

    for (;;)
{
    monitoring_count++;
    /* Monitoring/telemetry placeholder */
    vTaskDelay(pdMS_TO_TICKS(100));
}
}

/* Static task memory */
static StaticTask_t gnc_tcb;
static StaticTask_t sensing_tcb;
static StaticTask_t communications_tcb;
static StaticTask_t monitoring_tcb;

static StackType_t gnc_stack[configMINIMAL_STACK_SIZE];
static StackType_t sensing_stack[configMINIMAL_STACK_SIZE];
static StackType_t communications_stack[configMINIMAL_STACK_SIZE];
static StackType_t monitoring_stack[configMINIMAL_STACK_SIZE];

int main(void)
{
    xTaskCreateStatic(
        GNC_Task,
        "GNC",
        configMINIMAL_STACK_SIZE,
        NULL,
        4,
        gnc_stack,
        &gnc_tcb
    );

    xTaskCreateStatic(
        Sensing_Task,
        "SENSE",
        configMINIMAL_STACK_SIZE,
        NULL,
        3,
        sensing_stack,
        &sensing_tcb
    );

    xTaskCreateStatic(
        Communications_Task,
        "COMMS",
        configMINIMAL_STACK_SIZE,
        NULL,
        2,
        communications_stack,
        &communications_tcb
    );

    xTaskCreateStatic(
        Monitoring_Task,
        "MON",
        configMINIMAL_STACK_SIZE,
        NULL,
        1,
        monitoring_stack,
        &monitoring_tcb
    );

    vTaskStartScheduler();

    /* Scheduler should not return. */
    for (;;)
    {
    }
}