#include "arena.h"
#include <string.h>

mem_arena* arena_create(u64 reserve_size, u64 commit_size) {
    u32 page_size = plat_get_pagesize();

    reserve_size = ALIGN_UP_POW2(reserve_size, page_size);
    commit_size = ALIGN_UP_POW2(commit_size, page_size);

    mem_arena* arena = plat_mem_reserve(reserve_size);

    if (!plat_mem_commit(arena, commit_size)) { return NULL; }

    arena->reserve_size = reserve_size;
    arena->commit_size = commit_size;
    arena->pos = ARENA_BASE_POS;
    arena->commit_pos = commit_size;

    return arena;
}

void arena_destroy(mem_arena* arena) {
    plat_mem_release(arena, arena->reserve_size);
}

void* arena_push(mem_arena* arena, u64 size, bool nonzero) {
    // Align the memory first to the nearest sizeof(void*)
    // CPUs read memory in chunks--usually 4-byte or 8-byte blocks (words)
    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);

    u64 new_pos = pos_aligned + size;

    // Check that the new_pos is within the reserved limits
    if (new_pos > arena->reserve_size) { return NULL; }

    // If within reserved limits but not enough commit size, make new commit
    if (new_pos > arena->commit_pos) {
        u64 new_commit_pos = new_pos;
        new_commit_pos += arena->commit_size - 1;
        new_commit_pos -= new_commit_pos % arena->commit_size;
        new_commit_pos = MIN(new_commit_pos, arena->reserve_size);

        u8* mem = (u8*)arena + arena->commit_pos;
        u64 commit_size = new_commit_pos - arena->commit_pos;

        // Increase the commit size (note that mem is the old commit position in the arena)
        if (!plat_mem_commit(mem, commit_size)) { return NULL; }

        arena->commit_pos = new_commit_pos;
    }

    arena->pos = new_pos;

    u8* out = (u8*)arena + pos_aligned;

    if (!nonzero) { memset(out, 0, size); }

    return out;
}

void* arena_pop(mem_arena* arena, u64 size) {
    size = MIN(size, arena->pos - ARENA_BASE_POS);
    arena->pos -= size;
}

void* arena_pop_to(mem_arena* arena, u64 pos) {
    u64 size = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, size);
}

void* arena_clear(mem_arena* arena) {
    arena_pop_to(arena, ARENA_BASE_POS);
}

static _Thread_local mem_arena* _scratch_arenas[2] = { NULL, NULL };

mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts) {
    i32 scratch_index = -1;

    for (i32 i = 0; i < 2; i++) {
        bool conflict_found = false;

        for (u32 j = 0; j < num_conflicts; j++) {
            if (_scratch_arenas[i] == conflicts[j]) {
                conflict_found = true;
                break;
            }
        }

        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    // If conflicts are in both scratch arenas
    // User themselves needs to check for this case
    if (scratch_index == -1) return (mem_arena_temp){ 0 };

    mem_arena** selected = &_scratch_arenas[scratch_index];

    if (*selected == NULL) *selected = arena_create(MiB(64), MiB(1));

    return (mem_arena_temp) {
        .arena = *selected,
        .start_pos = (*selected)->pos
    };
}

void arena_scratch_release(mem_arena_temp scratch) {
    arena_pop_to(scratch.arena, scratch.start_pos);
}

#ifdef _WIN32
#include <windows.h>

u32 plat_get_pagesize(void) {
    SYSTEM_INFO sys_info = { 0 };
    GetSystemInfo(&sys_info);

    return sys_info.dwPageSize;
}

void* plat_mem_reserve(u64 size) {
    // Tell the OS to set aside a contiguous block of virtual addresses for virtual use
    // No physical RAM or page file is consumed during this step
    // Ensures that as the arena grows, it remains perfectly contiguous in memory

    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_READWRITE);
}

bool plat_mem_commit(void* ptr, u64 size) {
    // The OS allocates physical memory or pagefile space to back those virtual addresses
    // As the arena allocator's internal offset pointer moves forward and crosses a page boundary
    // it dynamically commits the next page

    void* ret = VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
    return ret != NULL;
}

bool plat_mem_decommit(void* ptr, u64 size) {
    // Tell the OS to free up the physical RAM but keep the addresses reserved
    // The memory becomes inaccessible but nobody else can take that virtual address space

    BOOL ret = VirtualFree(ptr, size, MEM_DECOMMIT);
    return ret;
}

bool plat_mem_release(void* ptr, u64 size) {
    // The virtual address space is freed and any remaining physical RAM backing is returned

    return VirtualFree(ptr, size, MEM_RELEASE);
}
#endif