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
    if (scanf("%f %f %f %f %f %f %d %d %f %f %f %f %f %f %f %f",
              &p.cell.scale_R, &p.cell.scale_Q, &p.cell.scale_plate,
              &p.cell.Rfac, &p.cell.Qloss, &p.cell.hA,
              &p.S, &p.P, &p.u_max, &p.V_max, &p.T_max,
              &p.dV, &p.dT, &p.dP, &dt, &w) != 16) return 1;

    zg_state_t s;
    float u = 0.0f, sink = 0.0f;
    while (scanf("%f %f %f", &s.soc, &s.T, &s.V1) == 3) {
        struct timespec a, b;
        zg_status_t st = zg_limit(&p, &s, dt, w, &u);      /* warm the caches */
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < REPS; ++i) { st = zg_limit(&p, &s, dt, w, &u); sink += u; }
        clock_gettime(CLOCK_MONOTONIC, &b);
        const double ns = ((double)(b.tv_sec - a.tv_sec) * 1e9
                           + (double)(b.tv_nsec - a.tv_nsec)) / (double)REPS;
        printf("%d %.1f\n", (int)st, ns);
    }
    return (sink == 12345.678f) ? 1 : 0;                   /* keep the loop from vanishing */
}
