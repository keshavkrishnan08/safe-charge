/* Test harness: read states on stdin, write the certificate's answer on stdout.
 *
 * This exists so the firmware can be diffed against the Python reference on the same states,
 * rather than being trusted. It is not part of the deployable image and e14_firmware.py
 * measures the library's footprint, not this file's.
 */
#include "zeroguard.h"
#include <stdio.h>

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
    float u;
    while (scanf("%f %f %f", &s.soc, &s.T, &s.V1) == 3) {
        const zg_status_t st = zg_limit(&p, &s, dt, w, &u);
        printf("%d %.9g\n", (int)st, (double)u);
    }
    return 0;
}
