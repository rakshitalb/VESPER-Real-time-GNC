#include <stdint.h>

extern int main(void);

extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

void Reset_Handler(void);
void Default_Handler(void);

__attribute__((section(".isr_vector")))
const uint32_t vector_table[] =
{
    0x20020000,
    (uint32_t)Reset_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    0,
    0,
    0,
    0,
    (uint32_t)Default_Handler
};

void Reset_Handler(void)
{
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    while (dst < &_edata)
    {
        *dst++ = *src++;
    }

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

void Default_Handler(void)
{
    while (1)
    {
    }
}