#include "rand.h"

void prng_seed_r(prng_state* rng, u64 init_state, u64 init_seq) {
    rng->state = 0U;
    rng->inc = (init_seq << 1u) | 1u;
    prng_rand_r(rng);
    rng->state += init_state;
    prng_rand_r(rng);

    rng->prev_norm = NAN;
}

void prng_seed(u64 init_state, u64 init_seq) {
    prng_seed_r(&s_prng_state, init_state, init_seq);
}

u32 prng_rand_r(prng_state* rng) {
    u64 oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

u32 prng_rand(void) {
    prng_rand_r(&s_prng_state);
}

f32 prng_randf_r(prng_state* rng) {
    // prng_rand_r is uniformly distributed along UINT32_MAX, so the division normalizes it
    return (f32)prng_rand_r(rng) / (f32)UINT32_MAX;
}

f32 prng_randf(void) {
    return prng_randf_r(&s_prng_state);
}

f32 prng_rand_norm_r(prng_state* rng) {
    if (!isnan(rng->prev_norm)) {
        f32 out = rng->prev_norm;
        rng->prev_norm = NAN;
        return out;
    }

    // This utilizes the Box-Muller transform
    // This transform utilizes the interval [0, 1] which is perfect for using prng_randf_r
    // which generates on this interval
    f32 u1 = 0.0f;
    do {
        u1 = prng_randf_r(rng);
    } while (u1 == 0.0f);

    f32 u2 = prng_randf_r(rng);
    f32 r = sqrtf(-2.0 * logf(u1));
    f32 z0 = r*cosf(2 * M_PI * u2), z1 = r * sinf(2 * M_PI * u2);

    rng->prev_norm = z1;
    return z0;
}

f32 prng_rand_norm(void) {
    return prng_rand_norm_r(&s_prng_state);
}

#ifdef _WIN32
void plat_get_entropy(void* data, u32 size) {
    BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}
#endif