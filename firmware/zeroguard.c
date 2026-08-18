/* zeroguard.c -- the two-sided safety certificate, as firmware.
 *
 * E13 measured what this *would* cost on a battery-management microcontroller: bytes of state,
 * bytes of table, whether the search needs an FPU. Those were estimates made from a Python
 * model of the arithmetic. This is the thing itself, so the estimate can be replaced by a
 * binary that compiles, runs, and can be diffed against the reference implementation.
 *
 * Two design choices carry over from E13 and both are about which way an error can point.
 *
 *   The search is integer. The current is a count of quanta, u = q * (hi - lo) / 2^BITS, and
 *   the bisection's only arithmetic is (lo + hi) >> 1. An integer midpoint rounds *down*, so
 *   the returned current is at or below the real-valued answer -- reduced resolution costs
 *   charge and cannot cost the guarantee. No FPU is needed for the part that decides.
 *
 *   The model is float32. A Cortex-M4F has a single-precision FPU and no double, so double
 *   would be emulated. E13 found single precision is *not* one-sided -- it was less
 *   conservative than double in 2 672 of 20 000 states -- which is why the margin exists and
 *   why e14_firmware.py measures the resulting deviation against it rather than assuming it
 *   away.
 *
 * Both directions are implemented. On charge the constraints are all caps and one bisection
 * suffices -- ZG_BITS + 2 evaluations. On discharge the load is a *floor*: the pack must deliver
 * it or the vehicle is not doing its job, and a second bisection finds that edge, for
 * 2*ZG_BITS + 4. That second family is the paper's contribution, so shipping only the first
 * would have been shipping the part that was already published.
 *
 * Constraints of the standards this is meant to satisfy, met by construction rather than by
 * inspection: no heap, no recursion, no variable-length structure, no unbounded loop. Every
 * loop bound is a compile-time constant, so worst-case execution time is a property of the
 * source rather than of the data.
 */
#include "zeroguard.h"
#include "zg_params.h"
#include <math.h>

#define ZG_BITS 16                 /* bisection resolution: u_max / 65536 */

/* ---------------------------------------------------------------------------------------
 * Open-circuit voltage: uniform table, linear interpolation. The table is generated from the
 * same file the Python reference loads, so the two cannot disagree about the cell.
 * ------------------------------------------------------------------------------------- */
static float zg_ocv_of(float soc)
{
    if (soc <= 0.0f) return zg_ocv[0];
    if (soc >= 1.0f) return zg_ocv[ZG_N_OCV - 1];
    const float x = soc * (float)(ZG_N_OCV - 1);
    const int   i = (int)x;
    const float f = x - (float)i;
    return zg_ocv[i] + f * (zg_ocv[i + 1] - zg_ocv[i]);
}

/* ---------------------------------------------------------------------------------------
 * One-step prediction. Mirrors Cell.probe in zeroguard/platforms.py term for term, in the
 * same order, because a different summation order is a different rounding and the whole point
 * of E14 is to be able to compare against the reference without an asterisk.
 * ------------------------------------------------------------------------------------- */
void zg_probe(const zg_cell_t *c, const zg_state_t *s, float I, float dt, float w,
              zg_out_t *out)
{
    const float Tk    = s->T + 273.15f;
    const float Rohm  = fabsf(ZG_A0 + ZG_A3 * expf(6.0f * (s->soc - 1.0f)))
                      * expf(ZG_EA * (1.0f / Tk - 1.0f / ZG_TREF)) * c->scale_R;
    float i0 = ZG_I0_0 * expf(ZG_I0_1 * s->soc);
    if (i0 < 1e-3f)  i0 = 1e-3f;
    if (i0 > 50.0f)  i0 = 50.0f;

    const float ohm = I * Rohm * c->Rfac;
    const float act = ZG_AACT * (1.0f + ZG_CT * (s->T - 25.0f))
                    * asinhf(I / (2.0f * i0)) * c->Rfac;

    float R1 = ZG_R1_FRAC * (Rohm * c->Rfac);
    if (R1 < 1e-4f) R1 = 1e-4f;
    const float a   = expf(-dt / ZG_TAU1);              /* zero-order hold, rc="exact" */
    const float V1n = a * s->V1 + (1.0f - a) * I * R1;

    const float eta = ohm + act + V1n;
    out->V  = zg_ocv_of(s->soc) + eta;

    const float R1c   = (R1 > 1e-6f) ? R1 : 1e-6f;
    const float Q_gen = I * ohm + I * act + V1n * V1n / R1c + I * Tk * ZG_DUDT;
    out->T  = s->T + dt * (Q_gen - c->hA * (s->T - w)) / ZG_C_TH;

    const float Qeff = ZG_Q_NOM * c->scale_Q * (1.0f - c->Qloss);
    out->soc = s->soc + I * dt / (3600.0f * Qeff);

    const float p0 = ZG_PL_P0A + ZG_PL_P0B * s->soc + ZG_PL_P0C * s->soc * s->soc
                   + ZG_PL_P0D * expf(6.0f * (s->soc - 1.0f));
    const float p1 = (ZG_PL_P1A + ZG_PL_P1B * s->soc + ZG_PL_P1C * s->soc * s->soc)
                   * expf(ZG_PL_ET * (1.0f / Tk - 1.0f / ZG_TREF));
    out->phi = (p0 - p1 * I) / c->scale_plate;
    out->V1  = V1n;
}

/* Every declared cap, with its margin folded in. Charge mode only: voltage, temperature and
 * the plating floor, which bounds current from above despite being written as `>=`. That is
 * the distinction the reference calls the constraint's *side*, and it is declared here for the
 * same reason it is declared there -- it does not follow from the comparison sense. */
static int zg_caps_ok(const zg_pack_t *p, const zg_out_t *o)
{
    if (!(o->T <= p->T_max - p->dT)) return 0;
    if (p->mode == ZG_CHARGE) {
        if (!(o->V   <= p->V_max - p->dV))        return 0;
        if (!(o->phi >= ZG_PLATE_MARGIN + p->dP)) return 0;
    } else {
        /* On discharge the cell sags: voltage and state of charge are floors on the *signal*
         * and caps on the *current*, which is the distinction the reference calls a
         * constraint's side. It does not follow from the comparison sense. */
        if (!(o->V   >= p->V_min     + p->dV))    return 0;
        if (!(o->soc >= p->soc_floor + p->dSoc))  return 0;
    }
    return 1;
}

/* Every constraint that bounds u from *below*: the load must be met. Discharge only. */
static int zg_floors_ok(const zg_pack_t *p, const zg_out_t *o, float u)
{
    if (p->mode != ZG_DISCHARGE) return 1;
    const float P_elec = (float)p->S * o->V * u;
    return (P_elec >= p->load_W + p->dLoad);
}

/* Pack current to cell current. The sign is the whole difference between the two modes and it
 * is easy to drop: positive charges the cell, so a *discharge* of u amps is -u/P into the model.
 * Omitting the flip compiles, runs, and disagrees with the reference on a seventh of all states
 * -- which is how it was found. */
static void zg_probe_pack(const zg_pack_t *p, const zg_state_t *s, float u, float dt, float w,
                          zg_out_t *o)
{
    const float i_cell = u / (float)p->P;
    zg_probe(&p->cell, s, (p->mode == ZG_CHARGE) ? i_cell : -i_cell, dt, w, o);
}

/* The temperature-dependent plating ceiling, and the actuator's own limit. */
static float zg_hi_bound(const zg_pack_t *p, const zg_state_t *s)
{
    if (p->mode == ZG_DISCHARGE) return p->u_max;   /* plating does not cap a discharge */
    float crate = 1.00f + 0.067f * (s->T - 10.0f);
    if (crate < 0.70f) crate = 0.70f;
    if (crate > 2.00f) crate = 2.00f;
    const float cap = crate * (ZG_Q_NOM * p->cell.scale_Q) * (float)p->P;
    float hi = (p->u_max < cap) ? p->u_max : cap;
    return (hi > 0.0f) ? hi : 0.0f;
}

/* ---------------------------------------------------------------------------------------
 * The certificate. Returns the largest admissible charge current, or 0 with status
 * ZG_INFEASIBLE when the state is already outside the safe set and no input recovers it in
 * one step -- which is a distinct condition from "the answer is zero" and is reported as one.
 *
 * Exactly ZG_BITS + 2 model evaluations, always. Not on average.
 * ------------------------------------------------------------------------------------- */
zg_status_t zg_limit(const zg_pack_t *p, const zg_state_t *s, float dt, float w, float *u_out)
{
    zg_out_t o;
    const float hi_b = zg_hi_bound(p, s);

    zg_probe_pack(p, s, 0.0f, dt, w, &o);                 /* eval 1 */
    if (!zg_caps_ok(p, &o)) { *u_out = 0.0f; return ZG_INFEASIBLE; }

    zg_probe_pack(p, s, hi_b, dt, w, &o);                 /* eval 2 */
    if (zg_caps_ok(p, &o)) { *u_out = hi_b; return ZG_OK; }

    /* Integer bisection over quanta of hi_b. `lo` is always cap-feasible and `hi` never is,
     * so the invariant is maintained by construction and the answer is `lo` -- the largest
     * quantum known to be safe, never an interpolation between a safe and an unsafe one. */
    uint32_t lo = 0u, hi = 1u << ZG_BITS;
    const float q = hi_b / (float)(1u << ZG_BITS);
    for (int k = 0; k < ZG_BITS; ++k) {                   /* evals 3 .. ZG_BITS+2 */
        const uint32_t mid = (lo + hi) >> 1;              /* rounds down: conservative */
        zg_probe_pack(p, s, (float)mid * q, dt, w, &o);
        if (zg_caps_ok(p, &o)) lo = mid; else hi = mid;
    }
    *u_out = (float)lo * q;
    return ZG_OK;
}

/* The two-sided form. The floor search is bounded by u_hi, not u_max: delivered power is
 * S*V(u)*u with V falling in u, so it peaks at the maximum-power point and falls beyond it.
 * The voltage cap keeps the search on the rising branch, where the feasible set is a suffix. */
zg_status_t zg_interval(const zg_pack_t *p, const zg_state_t *s, float dt, float w,
                        float *lo_out, float *hi_out)
{
    float u_hi;
    const zg_status_t st = zg_limit(p, s, dt, w, &u_hi);
    if (st != ZG_OK) { *lo_out = *hi_out = 0.0f; return st; }
    *hi_out = u_hi;

    if (p->mode == ZG_CHARGE) { *lo_out = 0.0f; return ZG_OK; }

    zg_out_t o;
    zg_probe_pack(p, s, 0.0f, dt, w, &o);
    if (zg_floors_ok(p, &o, 0.0f)) { *lo_out = 0.0f; return ZG_OK; }
    zg_probe_pack(p, s, u_hi, dt, w, &o);
    if (!zg_floors_ok(p, &o, u_hi)) {           /* the load costs more than the caps allow */
        *lo_out = u_hi; return ZG_INFEASIBLE;
    }
    uint32_t lo = 0u, hi = 1u << ZG_BITS;
    const float q = u_hi / (float)(1u << ZG_BITS);
    for (int k = 0; k < ZG_BITS; ++k) {
        const uint32_t mid = (lo + hi) >> 1;
        const float um = (float)mid * q;
        zg_probe_pack(p, s, um, dt, w, &o);
        if (zg_floors_ok(p, &o, um)) hi = mid; else lo = mid;
    }
    *lo_out = (float)hi * q;                     /* round *up*: the load is met, conservatively */
    return ZG_OK;
}

zg_status_t zg_project(const zg_pack_t *p, const zg_state_t *s, float u_req, float dt, float w,
                       float *u_out)
{
    float u_hi;
    const zg_status_t st = zg_limit(p, s, dt, w, &u_hi);
    if (st != ZG_OK) { *u_out = 0.0f; return st; }
    if (u_req < 0.0f)  u_req = 0.0f;
    *u_out = (u_req < u_hi) ? u_req : u_hi;
    return ZG_OK;
}
