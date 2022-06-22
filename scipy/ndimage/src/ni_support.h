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
#define NI_ITERATOR_RESET(_it)                                                \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = 0; _ii <= (_it).rank_m1; _ii++)                                 \
        (_it).coordinates[_ii] = 0;                                           \
}

/* go to the next point in a single array */
#define NI_ITERATOR_NEXT(_it, _ptr)                                           \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it).rank_m1; _ii >= 0; _ii--)                                 \
        if ((_it).coordinates[_ii] < (_it).dimensions[_ii]) {                 \
            (_it).coordinates[_ii]++;                                         \
            _ptr += (_it).strides[_ii];                                       \
            break;                                                            \
        } else {                                                              \
            (_it).coordinates[_ii] = 0;                                       \
            _ptr -= (_it).backstrides[_ii];                                   \
        }                                                                     \
}

/* go to the next point in two arrays of the same size */
#define NI_ITERATOR_NEXT2(_it1, _it2, _ptr1, _ptr2)                           \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it1).rank_m1; _ii >= 0; _ii--)                                \
        if ((_it1).coordinates[_ii] < (_it1).dimensions[_ii]) {               \
            (_it1).coordinates[_ii]++;                                        \
            _ptr1 += (_it1).strides[_ii];                                     \
            _ptr2 += (_it2).strides[_ii];                                     \
            break;                                                            \
        } else {                                                              \
            (_it1).coordinates[_ii] = 0;                                      \
            _ptr1 -= (_it1).backstrides[_ii];                                 \
            _ptr2 -= (_it2).backstrides[_ii];                                 \
        }                                                                     \
}

/* go to the next point in three arrays of the same size */
#define NI_ITERATOR_NEXT3(_it1, _it2, _it3, _ptr1, _ptr2, _ptr3)              \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it1).rank_m1; _ii >= 0; _ii--)                                \
        if ((_it1).coordinates[_ii] < (_it1).dimensions[_ii]) {               \
            (_it1).coordinates[_ii]++;                                        \
            _ptr1 += (_it1).strides[_ii];                                     \
            _ptr2 += (_it2).strides[_ii];                                     \
            _ptr3 += (_it3).strides[_ii];                                     \
            break;                                                            \
        } else {                                                              \
            (_it1).coordinates[_ii] = 0;                                      \
            _ptr1 -= (_it1).backstrides[_ii];                                 \
            _ptr2 -= (_it2).backstrides[_ii];                                 \
            _ptr3 -= (_it3).backstrides[_ii];                                 \
        }                                                                     \
}

/* go to an arbitrary point in a single array */
#define NI_ITERATOR_GOTO(_it, _dest, _base, _ptr)                             \
{                                                                             \
    int _ii;                                                                  \
    _ptr = _base;                                                             \
    for(_ii = (_it).rank_m1; _ii >= 0; _ii--) {                               \
        _ptr += _dest[_ii] * (_it).strides[_ii];                              \
        (_it).coordinates[_ii] = _dest[_ii];                                  \
    }                                                                         \
}

/******************************************************************/
/* Line buffers */
/******************************************************************/

/* the linebuffer structure: */
typedef struct {
    double *buffer_data;
    npy_intp buffer_size;  /* bytes */
    npy_intp buffer_lines, line_length, line_stride;
    npy_intp size1, size2, array_lines, next_line;
    NI_Iterator iterator;
    char* array_data;
    npy_intp array_size;  /* bytes */
    enum NPY_TYPES array_type;
    NI_ExtendMode extend_mode;
    double extend_value;
} NI_LineBuffer;

/* Get the next line being processed: */
#define NI_GET_LINE(_buffer, _line)                                           \
    ((_buffer).buffer_data + (_line) * ((_buffer).line_length +               \
     (_buffer).size1 + (_buffer).size2))
/* Allocate line buffer data */
int NI_AllocateLineBuffer(PyArrayObject*, int, npy_intp, npy_intp,
                           npy_intp*, npy_intp, double**, npy_intp*);

/* Initialize a line buffer */
int NI_InitLineBuffer(PyArrayObject*, int, npy_intp, npy_intp, npy_intp,
                      double*, npy_intp, NI_ExtendMode, double, NI_LineBuffer*);

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
#define NI_FILTER_NEXT(_itf, _it1, _ptrf, _ptr1)                              \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it1).rank_m1; _ii >= 0; _ii--) {                              \
        npy_intp _pp = (_it1).coordinates[_ii];                               \
        if (_pp < (_it1).dimensions[_ii]) {                                   \
            if (_pp < (_itf).bound1[_ii] ||                                   \
                _pp >= (_itf).bound2[_ii])                                    \
                _ptrf += (_itf).strides[_ii];                                 \
            (_it1).coordinates[_ii]++;                                        \
            _ptr1 += (_it1).strides[_ii];                                     \
            break;                                                            \
        } else {                                                              \
            (_it1).coordinates[_ii] = 0;                                      \
            _ptr1 -= (_it1).backstrides[_ii];                                 \
            _ptrf -= (_itf).backstrides[_ii];                                 \
        }                                                                     \
    }                                                                         \
}

/* Move to the next point in two arrays, possible changing the pointer
     to the filter offsets when moving into a different region in the
     array: */
#define NI_FILTER_NEXT2(_itf, _it1, _it2, _ptrf, _ptr1, _ptr2)                \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it1).rank_m1; _ii >= 0; _ii--) {                              \
        npy_intp _pp = (_it1).coordinates[_ii];                               \
        if (_pp < (_it1).dimensions[_ii]) {                                   \
            if (_pp < (_itf).bound1[_ii] ||                                   \
                _pp >= (_itf).bound2[_ii])                                    \
                _ptrf += (_itf).strides[_ii];                                 \
            (_it1).coordinates[_ii]++;                                        \
            _ptr1 += (_it1).strides[_ii];                                     \
            _ptr2 += (_it2).strides[_ii];                                     \
            break;                                                            \
        } else {                                                              \
            (_it1).coordinates[_ii] = 0;                                      \
            _ptr1 -= (_it1).backstrides[_ii];                                 \
            _ptr2 -= (_it2).backstrides[_ii];                                 \
            _ptrf -= (_itf).backstrides[_ii];                                 \
        }                                                                     \
    }                                                                         \
}

/* Move to the next point in three arrays, possible changing the pointer
     to the filter offsets when moving into a different region in the
     array: */
#define NI_FILTER_NEXT3(_itf, _it1, _it2, _it3, _ptrf, _ptr1, _ptr2, _ptr3)   \
{                                                                             \
    int _ii;                                                                  \
    for(_ii = (_it1).rank_m1; _ii >= 0; _ii--) {                              \
        npy_intp _pp = (_it1).coordinates[_ii];                               \
        if (_pp < (_it1).dimensions[_ii]) {                                   \
            if (_pp < (_itf).bound1[_ii] ||                                   \
                _pp >= (_itf).bound2[_ii])                                    \
                _ptrf += (_itf).strides[_ii];                                 \
            (_it1).coordinates[_ii]++;                                        \
            _ptr1 += (_it1).strides[_ii];                                     \
            _ptr2 += (_it2).strides[_ii];                                     \
            _ptr3 += (_it3).strides[_ii];                                     \
            break;                                                            \
        } else {                                                              \
            (_it1).coordinates[_ii] = 0;                                      \
            _ptr1 -= (_it1).backstrides[_ii];                                 \
            _ptr2 -= (_it2).backstrides[_ii];                                 \
            _ptr3 -= (_it3).backstrides[_ii];                                 \
            _ptrf -= (_itf).backstrides[_ii];                                 \
        }                                                                     \
    }                                                                         \
}

/* Move the pointer to the filter offsets according to the given
    coordinates: */
#define NI_FILTER_GOTO(_itf, _it, _fbase, _ptrf)                              \
{                                                                             \
    int _ii;                                                                  \
    npy_intp _jj;                                                             \
    _ptrf = _fbase;                                                           \
    for(_ii = _it.rank_m1; _ii >= 0; _ii--) {                                 \
        npy_intp _pp = _it.coordinates[_ii];                                  \
        npy_intp b1 = (_itf).bound1[_ii];                                     \
        npy_intp b2 = (_itf).bound2[_ii];                                     \
        if (_pp < b1) {                                                       \
            _jj = _pp;                                                        \
        } else if (_pp > b2 && b2 >= b1) {                                    \
            _jj = _pp + b1 - b2;                                              \
        } else {                                                              \
            _jj = b1;                                                         \
        }                                                                     \
        _ptrf += (_itf).strides[_ii] * _jj;                                   \
    }                                                                         \
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
