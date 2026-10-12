#ifndef RAND_H
#define RAND_H

#define _USE_MATH_DEFINES
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Based on PCG random number generation: https://www.pcg-random.org/

typedef struct {
    u64 state;  // RNG state--all values are possible
    u64 inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd

    f32 prev_norm;
} prng_state;

static prng_state s_prng_state = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL, NAN };

void prng_seed_r(prng_state* rng, u64 init_state, u64 init_seq);
void prng_seed(u64 init_state, u64 init_seq);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

f32 prng_rand_norm_r(prng_state* rng);
f32 prng_rand_norm(void);

#ifdef _WIN32
#include <Windows.h>
#include <bcrypt.h>
void plat_get_entropy(void* data, u32 size);
#endif

#endif