#include <stddef.h>
#include <stdint.h>

void *memset(void *dest, int value, size_t count)
{
    uint8_t *dst = (uint8_t *)dest;

    while (count--)
    {
        *dst++ = (uint8_t)value;
    }

    return dest;
}

void *memcpy(void *dest, const void *src, size_t count)
{
    uint8_t *dst = (uint8_t *)dest;
    const uint8_t *source = (const uint8_t *)src;

    while (count--)
    {
        *dst++ = *source++;
    }

    return dest;
}