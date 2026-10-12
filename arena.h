#ifndef ARENA_H
#define ARENA_H

#include "types.h"
#include <stdbool.h>

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

#define ARENA_BASE_POS (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void*))

typedef struct {
    u64 reserve_size;
    u64 commit_size;

    u64 pos;
    u64 commit_pos;
} mem_arena;

mem_arena* arena_create(u64 reserve_size, u64 commit_size);
void arena_destroy(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size, bool nonzero);
void* arena_pop(mem_arena* arena, u64 size);
void* arena_pop_to(mem_arena* arena, u64 pos);
void* arena_clear(mem_arena* arena);

typedef struct {
    mem_arena* arena;
    u64 start_pos;
} mem_arena_temp;

mem_arena_temp arena_temp_begin(mem_arena* arena);
void arena_temp_end(mem_arena_temp temp);

mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts);
void arena_scratch_release(mem_arena_temp scratch);

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)

#ifdef _WIN32
u32 plat_get_pagesize(void);
void* plat_mem_reserve(u64 size);            // Reserve bytes in virtual address space
bool plat_mem_commit(void* ptr, u64 size);   // Commit to that portion of virtual address space
bool plat_mem_decommit(void* ptr, u64 size);
bool plat_mem_release(void* ptr, u64 size);
#endif

#endif