/* zeroguard.h -- the charge-direction safety certificate for a battery management unit.
 *
 * No heap, no recursion, no variable-length structure, no unbounded loop. One call is a fixed
 * 18 one-step model evaluations and returns the largest admissible charge current.
 */
#ifndef ZEROGUARD_H
#define ZEROGUARD_H

#include <stdint.h>

typedef enum { ZG_OK = 0, ZG_INFEASIBLE = 1 } zg_status_t;

/* Per-cell parameters the estimator is allowed to be wrong about, held at the worst corner of
 * the declared envelope. These are the only numbers an integrator supplies. */
typedef struct {
    float scale_R;        /* series resistance multiple assumed, e.g. 1.8 */
    float scale_Q;        /* capacity multiple assumed, e.g. 0.80         */
    float scale_plate;    /* plating-proxy multiple assumed, e.g. 1.6     */
    float Rfac;           /* aging resistance factor (1.0 = fresh)        */
    float Qloss;          /* aging capacity loss  (0.0 = fresh)           */
    float hA;             /* cooling coefficient, W/K                     */
} zg_cell_t;

typedef struct {
    zg_cell_t cell;
    int32_t   S;          /* cells in series   */
    int32_t   P;          /* strings in parallel */
    float     u_max;      /* actuator ceiling, pack amps */
    float     V_max;      /* per-cell voltage limit */
    float     T_max;      /* cell temperature limit, degC */
    float     dV, dT, dP; /* margins: voltage, thermal, plating */
} zg_pack_t;

typedef struct { float soc; float T; float V1; } zg_state_t;
typedef struct { float V; float T; float phi; float soc; float V1; } zg_out_t;

void        zg_probe(const zg_cell_t *c, const zg_state_t *s, float I, float dt, float w,
                     zg_out_t *out);
zg_status_t zg_limit(const zg_pack_t *p, const zg_state_t *s, float dt, float w, float *u_out);
zg_status_t zg_project(const zg_pack_t *p, const zg_state_t *s, float u_req, float dt, float w,
                       float *u_out);

#endif
