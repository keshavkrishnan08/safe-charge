/* Timing harness. The footprint claim became a measurement in E14; the *timing* claim did not,
 * and was still coming from the Python reference -- which says nothing about a microcontroller.
 * This measures the compiled routine.
 *
 * What matters for a safety task slot is not the mean but the worst case, and whether the worst
 * case is bounded. `zg_limit` has exactly three paths: infeasible at the anchor (1 evaluation),
 * feasible at the ceiling (2), or the full bisection (ZG_BITS + 2). None depends on the data
 * beyond selecting among those three, so the worst case is a property of the source. This prints
 * per-state nanoseconds so that claim can be checked rather than believed.
 */
#include "zeroguard.h"
#include <stdio.h>
#include <time.h>

#define REPS 2000

int main(void)
{
    zg_pack_t p;
    float dt, w;
    int mode;
    if (scanf("%d %f %f %f %f %f %f %d %d %f %f %f %f %f %f %f %f %f %f %f %f",
              &mode, &p.cell.scale_R, &p.cell.scale_Q, &p.cell.scale_plate,
              &p.cell.Rfac, &p.cell.Qloss, &p.cell.hA,
              &p.S, &p.P, &p.u_max, &p.V_max, &p.T_max,
              &p.V_min, &p.soc_floor, &p.load_W,
              &p.dV, &p.dT, &p.dP, &p.dSoc, &p.dLoad, &dt) != 21) return 1;
    if (scanf("%f", &w) != 1) return 1;
    p.mode = (mode == 0) ? ZG_CHARGE : ZG_DISCHARGE;

    zg_state_t s;
    float u = 0.0f, sink = 0.0f;
    while (scanf("%f %f %f", &s.soc, &s.T, &s.V1) == 3) {
        struct timespec a, b;
        float lo;
        zg_status_t st = zg_interval(&p, &s, dt, w, &lo, &u);   /* warm the caches */
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < REPS; ++i) { st = zg_interval(&p, &s, dt, w, &lo, &u); sink += u; }
        clock_gettime(CLOCK_MONOTONIC, &b);
        const double ns = ((double)(b.tv_sec - a.tv_sec) * 1e9
                           + (double)(b.tv_nsec - a.tv_nsec)) / (double)REPS;
        printf("%d %.1f %.1f\n", (int)st, ns, ns);
    }
    return (sink == 12345.678f) ? 1 : 0;                   /* keep the loop from vanishing */
}
