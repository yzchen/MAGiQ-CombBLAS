#ifndef MAGIQ_UTILS_H
#define MAGIQ_UTILS_H

// CombBLAS independent definitions

// IndexType is used for index in CombBLAS sparse matrix and dense vectors,
// for dataset which has less than 2^32-1(= 4B) vertices, can change it to uint32_t,
// starting from lubm13B, uint64_t is needed
#define IndexType uint64_t
#define ElementType uint8_t

// // comparasion struct for qsort in result generation
typedef struct {
    IndexType a, b, c;
} Int3;

int compInt3A(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt3B(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt3C(const void *elem1, const void *elem2) {
    Int3 f = *((Int3 *) elem1);
    Int3 s = *((Int3 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

typedef struct {
    IndexType a, b, c, d;
} Int4;

int compInt4A(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt4B(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt4C(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

int compInt4D(const void *elem1, const void *elem2) {
    Int4 f = *((Int4 *) elem1);
    Int4 s = *((Int4 *) elem2);
    if (f.d > s.d) return 1;
    if (f.d < s.d) return -1;
    return 0;
}

typedef struct {
    IndexType a, b, c, d, e;
} Int5;

int compInt5A(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.a > s.a) return 1;
    if (f.a < s.a) return -1;
    return 0;
}

int compInt5B(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.b > s.b) return 1;
    if (f.b < s.b) return -1;
    return 0;
}

int compInt5C(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.c > s.c) return 1;
    if (f.c < s.c) return -1;
    return 0;
}

int compInt5D(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.d > s.d) return 1;
    if (f.d < s.d) return -1;
    return 0;
}

int compInt5E(const void *elem1, const void *elem2) {
    Int5 f = *((Int5 *) elem1);
    Int5 s = *((Int5 *) elem2);
    if (f.e > s.e) return 1;
    if (f.e < s.e) return -1;
    return 0;
}
// // comparasion struct for qsort in result generation

// comparasion function pointer array
// this comp function design will limit the maximum columns in the final table
// currently maximum columns is 5
// TODO : hard coded way to scale the comp is allowed but not efficient,
// TODO : find a way to support in a scalable way
int (*comp[15])(const void *, const void *);

#endif // MAGIQ_UTILS_H