# SAARTHI – Real-Time GNC Loop Under Emulation

## Project status
Embedded real-time GNC implementation using Python reference control, C port, ARM Cortex-M4 build, FreeRTOS tasks, and Renode emulation.

## Environment
- Python reference model: Python 3.12.3
- NumPy: 2.5.0
- SciPy: 1.18.1
- Target: ARM Cortex-M4
- ARM GNU Toolchain: arm-none-eabi-gcc 15.3.1
- Renode: 1.17.0
- Renode CPU model: PerformanceInMips 125

## Main artifacts
- control.py - Python reference controller
- dynamics.py - spacecraft dynamics
- c_port/control.c - embedded C controller
- c_port/gnc_rtos_gnc.elf - FreeRTOS GNC build
- c_port/gnc_week9_integration.elf - Week 9 integration build
- c_port/gnc_week11_final_measurement.elf - final timing measurement build
- c_port/week9_integration_repro.resc - relative-path reproduction script

## Verification
- Analytical limits: VERIFIED; torque saturation = +/-0.010 N.m and wheel momentum limit = 0.100 N.m.s.
- Cross-code check: VERIFIED; C/Renode and Python/reference outputs matched [-0.010000, +0.007000, -0.003300] N.m.
- Convergence: VERIFIED WITH QUALIFICATION; dt = 1, 0.5, 0.25, 0.125 s produced unchanged final attitude at reported output precision.
- Published reference: PARTIAL / QUALIFIED; published benchmark = 0.35 arcmin, digitized Figure 5.4 mean = 0.3911 arcmin, relative difference = 11.74%.

## Timing
The observed standalone controller maximum was 7.288 us at 125 MIPS. The 10 ms task deadline margin is 9.992712 ms. Week 11 measured the GNC activation period at 10 ms with 0 ms peak-to-peak jitter at the 1 ms timer resolution. Execution min/max were not interpreted as zero microseconds because the timer resolution was too coarse.

## Week 9 integration
The Aerospace GNC control law was ported into C and integrated with the RTOS. GNC_Task publishes timestamp, status and torque through GNC_Telemetry; Communications_Task consumes the interface. Runtime evidence recorded non-zero timestamps and valid status.

## Published-data limitation
The published target-tracking reference time series originated from an ARAPAIMA STK simulation and were not available as original numerical data. Figure 5.4 was therefore digitized from the published image. The digitized curve is not claimed to be the original simulation data.

## Reproducibility
Use c_port/week9_integration_repro.resc with gnc_week9_integration.elf for the clean relative-path Week 9 reproduction. Several legacy Renode scripts retain machine-specific absolute ELF paths and were intentionally not modified because they are verified artifacts. Therefore project-wide clone-and-run reproducibility remains a documented gap.

## Evidence files
- analytical_limits_results.txt
- convergence_results.txt
- four_way_verification_status.txt
- cross_domain_integration_status.txt
- reproducibility_status.txt
- published_reference_check/published_reference_evidence.txt
- c_port/week9_cross_code_results.txt
- c_port/week9_runtime_evidence.txt
- c_port/week11_final_measurement_results.txt

## Integrity principle
Results that could not be fully reproduced from available source data are explicitly marked as limitations rather than presented as successful reproductions.
