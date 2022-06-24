/* Copyright (C) 2003-2005 Peter J. Verveer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NI_SUPPORT_H
#define NI_SUPPORT_H

/*
 * The NO_ARRAY_IMPORT tells numpy that the compilation unit will reuse
 * the numpy API initialized in another compilation unit. The compilation
 * unit that initializes the shared numpy API by calling import_array()
 * must bypass this by explicitly including nd_image.h before ni_support.h.
 */
#define NO_IMPORT_ARRAY
#include "nd_image.h"
#undef NO_IMPORT_ARRAY

#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <assert.h>

/* The different boundary conditions. The mirror condition is not used
     by the python code, but C code is kept around in case we might wish
     to add it. */
typedef enum {
    NI_EXTEND_FIRST = 0,
    NI_EXTEND_NEAREST = 0,
    NI_EXTEND_WRAP = 1,
    NI_EXTEND_REFLECT = 2,
    NI_EXTEND_MIRROR = 3,
    NI_EXTEND_CONSTANT = 4,
    NI_EXTEND_GRID_WRAP = 5,
    NI_EXTEND_GRID_CONSTANT = 6,
    NI_EXTEND_LAST = NI_EXTEND_GRID_WRAP,
    NI_EXTEND_DEFAULT = NI_EXTEND_MIRROR
} NI_ExtendMode;


/******************************************************************/
/* Iterators */
/******************************************************************/

/* the iterator structure: */
typedef struct {
    int rank_m1;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXDIMS];
} NI_Iterator;

/* initialize iterations over single array elements: */
int NI_InitPointIterator(PyArrayObject*, NI_Iterator*);

/* initialize iterations over an arbritrary sub-space: */
int NI_SubspaceIterator(NI_Iterator*, npy_uint32);

/* initialize iteration over array lines: */
int NI_LineIterator(NI_Iterator*, int);

/* reset an iterator */
static inline void NI_IteratorReset(NI_Iterator *it)
{
    for (int ii = 0; ii <= it->rank_m1; ii++) {
        it->coordinates[ii] = 0;
    }
}

/* go to the next point in a single array */
static inline void NI_IteratorNext(NI_Iterator *it, void **ptr)
{
    for (int ii = it->rank_m1; ii >= 0; ii--) {
        if (it->coordinates[ii] < it->dimensions[ii]) {
            it->coordinates[ii]++;
            *ptr += it->strides[ii];
            break;

        } else {
            it->coordinates[ii] = 0;
            *ptr -= it->backstrides[ii];
        }
    }
}

/* go to the next point in two arrays of the same size */
static inline void NI_IteratorNext2(
    NI_Iterator *it1,
    NI_Iterator *it2,
    void **ptr1,
    void **ptr2)
{
    for (int ii = it1->rank_m1; ii >= 0; ii--) {
        if (it1->coordinates[ii] < it1->dimensions[ii]) {
            it1->coordinates[ii]++;
            *ptr1 += it1->strides[ii];
            *ptr2 += it2->strides[ii];
            break;

        } else {
            it1->coordinates[ii] = 0;
            *ptr1 -= it1->backstrides[ii];
            *ptr2 -= it2->backstrides[ii];
        }
    }
}

/* go to the next point in three arrays of the same size */
static inline void NI_IteratorNext3(
    NI_Iterator *it1,
    NI_Iterator *it2,
    NI_Iterator *it3,
    void **ptr1,
    void **ptr2,
    void **ptr3)
{
    for (int ii = it1->rank_m1; ii >= 0; ii--) {
        if (it1->coordinates[ii] < it1->dimensions[ii]) {
            it1->coordinates[ii]++;
            *ptr1 += it1->strides[ii];
            *ptr2 += it2->strides[ii];
            *ptr3 += it3->strides[ii];
            break;

        } else {
            it1->coordinates[ii] = 0;
            *ptr1 -= it1->backstrides[ii];
            *ptr2 -= it2->backstrides[ii];
            *ptr3 -= it3->backstrides[ii];
        }
    }
}

/* go to an arbitrary point in a single array */
static inline void NI_IteratorGoto(
    NI_Iterator *it,
    npy_intp *dest,
    void *base,
    void **ptr)
{
    *ptr = base;

    for (int ii = it->rank_m1; ii >= 0; ii--) {
        *ptr += dest[ii] * it->strides[ii];
        it->coordinates[ii] = dest[ii];
    }
}

/******************************************************************/
/* Line buffers */
/******************************************************************/

/* the linebuffer structure: */
typedef struct {
    double *buffer_data;
    npy_intp buffer_lines, line_length, line_stride;
    npy_intp size1, size2, array_lines, next_line;
    NI_Iterator iterator;
    char* array_data;
    enum NPY_TYPES array_type;
    NI_ExtendMode extend_mode;
    double extend_value;
} NI_LineBuffer;

/* Get the next line being processed: */
static inline double* NI_GetLine(NI_LineBuffer *buf, npy_intp line)
{
    return buf->buffer_data + line *
        (buf->line_length + buf->size1 + buf->size2);
}

/* Allocate line buffer data */
int NI_AllocateLineBuffer(PyArrayObject*, int, npy_intp, npy_intp,
                           npy_intp*, npy_intp, double**);

/* Initialize a line buffer */
int NI_InitLineBuffer(PyArrayObject*, int, npy_intp, npy_intp, npy_intp,
                                            double*, NI_ExtendMode, double, NI_LineBuffer*);

/* Extend a line in memory to implement boundary conditions: */
int NI_ExtendLine(double*, npy_intp, npy_intp, npy_intp, NI_ExtendMode, double);

/* Copy a line from an array to a buffer: */
int NI_ArrayToLineBuffer(NI_LineBuffer*, npy_intp*, int*);

/* Copy a line from a buffer to an array: */
int NI_LineBufferToArray(NI_LineBuffer*);

/******************************************************************/
/* Multi-dimensional filter support functions */
/******************************************************************/

/* the filter iterator structure: */
typedef struct {
    npy_intp strides[NPY_MAXDIMS], backstrides[NPY_MAXDIMS];
    npy_intp bound1[NPY_MAXDIMS], bound2[NPY_MAXDIMS];
} NI_FilterIterator;

/* Initialize a filter iterator: */
int NI_InitFilterIterator(int, npy_intp*, npy_intp, npy_intp*,
                          npy_intp*, NI_FilterIterator*);

/* Calculate the offsets to the filter points, for all border regions and
     the interior of the array: */
int NI_InitFilterOffsets(PyArrayObject*, npy_bool*, npy_intp*,
                         npy_intp*, NI_ExtendMode, npy_intp**,
                         npy_intp*, npy_intp**);

/* Move to the next point in an array, possible changing the filter
     offsets, to adapt to boundary conditions: */
static inline void NI_FilterNext(
    NI_FilterIterator *itf,
    NI_Iterator *it1,
    void ***ptrf,
    void **ptr1)
{
    for (int ii = it1->rank_m1; ii >= 0; ii--) {
        npy_intp pp = it1->coordinates[ii];

        if (pp < it1->dimensions[ii]) {
            if (pp < itf->bound1[ii] ||
                pp >= itf->bound2[ii])
            {
                *ptrf += itf->strides[ii];
            }
            it1->coordinates[ii]++;
            *ptr1 += it1->strides[ii];
            break;

        } else {
            it1->coordinates[ii] = 0;
            *ptr1 -= it1->backstrides[ii];
            *ptrf -= itf->backstrides[ii];
        }
    }
}

/* Move to the next point in two arrays, possible changing the pointer
     to the filter offsets when moving into a different region in the
     array: */
static inline void NI_FilterNext2(
    NI_FilterIterator *itf,
    NI_Iterator *it1,
    NI_Iterator *it2,
    void ***ptrf,
    void **ptr1,
    void **ptr2)
{
    for (int ii = it1->rank_m1; ii >= 0; ii--) {
        npy_intp pp = it1->coordinates[ii];

        if (pp < it1->dimensions[ii]) {
            if (pp < itf->bound1[ii] ||
                pp >= itf->bound2[ii])
            {
                *ptrf += itf->strides[ii];
            }
            it1->coordinates[ii]++;
            *ptr1 += it1->strides[ii];
            *ptr2 += it2->strides[ii];
            break;

        } else {
            it1->coordinates[ii] = 0;
            *ptr1 -= it1->backstrides[ii];
            *ptr2 -= it2->backstrides[ii];
            *ptrf -= itf->backstrides[ii];
        }
    }
}

/* Move to the next point in three arrays, possible changing the pointer
     to the filter offsets when moving into a different region in the
     array: */
static inline void NI_FilterNext3(
    NI_FilterIterator *itf,
    NI_Iterator *it1,
    NI_Iterator *it2,
    NI_Iterator *it3,
    void ***ptrf,
    void **ptr1,
    void **ptr2,
    void **ptr3)
{
    for (int ii = it1->rank_m1; ii >= 0; ii--) {
        npy_intp pp = it1->coordinates[ii];

        if (pp < it1->dimensions[ii]) {
            if (pp < itf->bound1[ii] ||
                pp >= itf->bound2[ii])
            {
                *ptrf += itf->strides[ii];
            }
            it1->coordinates[ii]++;
            *ptr1 += it1->strides[ii];
            *ptr2 += it2->strides[ii];
            *ptr3 += it3->strides[ii];
            break;

        } else {
            it1->coordinates[ii] = 0;
            *ptr1 -= it1->backstrides[ii];
            *ptr2 -= it2->backstrides[ii];
            *ptr3 -= it3->backstrides[ii];
            *ptrf -= itf->backstrides[ii];
        }
    }
}

/* Move the pointer to the filter offsets according to the given
    coordinates: */
static inline void NI_FilterGoto(
    NI_FilterIterator *itf,
    NI_Iterator *it,
    void *fbase,
    void **ptrf)
{
    npy_intp jj = 0;
    *ptrf = fbase;

    for (int ii = it->rank_m1; ii >= 0; ii--) {
        npy_intp pp = it->coordinates[ii];
        npy_intp b1 = itf->bound1[ii];
        npy_intp b2 = itf->bound2[ii];

        if (pp < b1) {
            jj = pp;
        } else if (pp > b2 && b2 >= b1) {
            jj = pp + b1 - b2;
        } else {
            jj = b1;
        }

        *ptrf += itf->strides[ii] * jj;
    }
}

typedef struct {
    npy_intp *coordinates;
        int size;
        void *next;
} NI_CoordinateBlock;

typedef struct {
        int block_size, rank;
        void *blocks;
} NI_CoordinateList;

NI_CoordinateList* NI_InitCoordinateList(int, int);
int NI_CoordinateListStealBlocks(NI_CoordinateList*, NI_CoordinateList*);
NI_CoordinateBlock* NI_CoordinateListAddBlock(NI_CoordinateList*);
NI_CoordinateBlock* NI_CoordinateListDeleteBlock(NI_CoordinateList*);
void NI_FreeCoordinateList(NI_CoordinateList*);

#endif
