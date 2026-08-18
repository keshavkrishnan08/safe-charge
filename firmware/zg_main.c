/* Test harness: read a pack description and states on stdin, write the certificate's answer.
 *
 * This exists so the firmware can be diffed against the Python reference on the same states
 * rather than trusted. It is not part of the deployable image, and e14_firmware.py measures the
 * library's footprint rather than this file's. Both directions are exercised: on charge the
 * lower edge is zero, on discharge it is the second bisection's answer.
 */
#include "zeroguard.h"
#include <stdio.h>

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
    float lo, hi;
    while (scanf("%f %f %f", &s.soc, &s.T, &s.V1) == 3) {
        const zg_status_t st = zg_interval(&p, &s, dt, w, &lo, &hi);
        printf("%d %.9g %.9g\n", (int)st, (double)lo, (double)hi);
    }
    return 0;
}
