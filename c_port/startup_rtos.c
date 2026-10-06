#include <stdint.h>

extern int main(void);

extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

/* FreeRTOS Cortex-M handlers */
extern void vPortSVCHandler(void);
extern void xPortPendSVHandler(void);
extern void xPortSysTickHandler(void);

void Reset_Handler(void);
void Default_Handler_Rtos(void);

__attribute__((section(".isr_vector")))
const uint32_t vector_table_rtos[] =
{
    /* Initial stack pointer */
    0x20020000,

    /* Cortex-M exception handlers */
    (uint32_t)Reset_Handler,   /* 1  Reset */
    (uint32_t)Default_Handler_Rtos, /* 2  NMI */
    (uint32_t)Default_Handler_Rtos, /* 3  HardFault */
    (uint32_t)Default_Handler_Rtos, /* 4  MemManage */
    (uint32_t)Default_Handler_Rtos, /* 5  BusFault */
    (uint32_t)Default_Handler_Rtos, /* 6  UsageFault */

    /* Reserved */
    0,
    0,
    0,
    0,

    /* FreeRTOS SVC */
    (uint32_t)vPortSVCHandler,      /* 11 SVC */

    /* Debug monitor */
    (uint32_t)Default_Handler_Rtos, /* 12 DebugMon */

    /* Reserved */
    0,

    /* FreeRTOS PendSV */
    (uint32_t)xPortPendSVHandler,   /* 14 PendSV */

    /* FreeRTOS SysTick */
    (uint32_t)xPortSysTickHandler  /* 15 SysTick */
};

void Reset_Handler(void)
{
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    /* Copy .data from Flash to RAM */
    while (dst < &_edata)
    {
        *dst++ = *src++;
    }

    /* Clear .bss */
    dst = &_sbss;

    while (dst < &_ebss)
    {
        *dst++ = 0;
    }

    main();

    while (1)
    {
    }
}

void Default_Handler_Rtos(void)
{
    while (1)
    {
    }
}