import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix

import pytest


def _check_csr_rowslice(i, sl, X, Xcsr):
    np_slice = X[i, sl]
    csr_slice = Xcsr[i, sl]
    assert_array_almost_equal(np_slice, csr_slice.toarray()[0])
    assert_(type(csr_slice) is csr_matrix)


def test_csr_rowslice():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    slices = [slice(None, None, None),
              slice(None, None, -1),
              slice(1, -2, 2),
              slice(-2, 1, -2)]

    for i in range(N):
        for sl in slices:
            _check_csr_rowslice(i, sl, X, Xcsr)


def test_csr_getrow():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_row = X[i:i + 1, :]
        csr_row = Xcsr.getrow(i)

        assert_array_almost_equal(arr_row, csr_row.toarray())
        assert_(type(csr_row) is csr_matrix)


def test_csr_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_col = X[:, i:i + 1]
        csr_col = Xcsr.getcol(i)

        assert_array_almost_equal(arr_col, csr_col.toarray())
        assert_(type(csr_col) is csr_matrix)

@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      0, (0, 4)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      1, (3, 0)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      'both', (0, 0)),
     (csr_matrix([[0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0]]),
      0, (0, 5))])
def test_csr_empty_slices(matrix_input, axis, expected_shape):
    # see gh-11127 for related discussion
    slice_1 = matrix_input.A.shape[0] - 1
    slice_2 = slice_1
    slice_3 = slice_2 - 1

    if axis == 0:
        actual_shape_1 = matrix_input[slice_1:slice_2, :].A.shape
        actual_shape_2 = matrix_input[slice_1:slice_3, :].A.shape
    elif axis == 1:
        actual_shape_1 = matrix_input[:, slice_1:slice_2].A.shape
        actual_shape_2 = matrix_input[:, slice_1:slice_3].A.shape
    elif axis == 'both':
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].A.shape
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].A.shape

    assert actual_shape_1 == expected_shape
    assert actual_shape_1 == actual_shape_2


def test_csr_bool_indexing():
    data = csr_matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    list_indices1 = [False, True, False]
    array_indices1 = np.array(list_indices1)
    list_indices2 = [[False, True, False], [False, True, False], [False, True, False]]
    array_indices2 = np.array(list_indices2)
    list_indices3 = ([False, True, False], [False, True, False])
    array_indices3 = (np.array(list_indices3[0]), np.array(list_indices3[1]))
    slice_list1 = data[list_indices1].toarray()
    slice_array1 = data[array_indices1].toarray()
    slice_list2 = data[list_indices2]
    slice_array2 = data[array_indices2]
    slice_list3 = data[list_indices3]
    slice_array3 = data[array_indices3]
    assert (slice_list1 == slice_array1).all()
    assert (slice_list2 == slice_array2).all()
    assert (slice_list3 == slice_array3).all()


# https://github.com/scipy/scipy/issues/9253
# error case
def test_csr_non_monotonic_indptr():
    with pytest.raises(
            ValueError,
            match="index pointer values must form a non-decreasing sequence"):
        m = csr_matrix(
            ([33, 44, 55], [0, 1, 2], [0, 3, 0, 0]),
            dtype=np.int8,
            shape=(3,3))
        m.toarray()


# https://github.com/scipy/scipy/issues/9253
# ok case
def test_csr_monotonic_indptr():
    m = csr_matrix(
        ([33, 44, 55], [0, 1, 2], [0, 3, 3, 3]),
        dtype=np.int8,
        shape=(3,3))
    assert m.toarray().tolist() == [[33, 44, 55], [0, 0, 0], [0, 0, 0]]


# https://github.com/scipy/scipy/issues/8778
# error case (oob)
def test_csr_oob_indices():
    with pytest.raises(
            ValueError,
            match="column index values must be < 1000"):
        data = [1.0, 1.0]
        indices = [1001, 555]
        indptr = [0, 1, 2]
        shape = (2, 1000)

        m = csr_matrix((data, indices, indptr), shape=shape)
        res = m * m.T
        res.toarray().tolist()


# https://github.com/scipy/scipy/issues/8778#issuecomment-787603693
# error case (negative indices)
def test_csr_neg_indices():
    with pytest.raises(
            ValueError,
            match="column index values must be >= 0"):
        data = [1.0, 1.0]
        # negative indices cause the program to write to memory at a lower address
        # than the vulnerable buffer
        indices = [-100, -555]
        indptr = [0, 1, 2]
        shape = (2, 1000)
        m = csr_matrix((data, indices, indptr), shape=shape)
        res = m * m.T
        res.toarray().tolist()


# https://github.com/scipy/scipy/issues/8778
# ok case
def test_csr_ok_indices():
    data = [1.0, 1.0]
    indices = [999, 555]
    indptr = [0, 1, 2]
    shape = (2, 1000)

    m = csr_matrix((data, indices, indptr), shape=shape)
    res = m * m.T
    assert res.toarray().tolist() == [[1.0, 0.0], [0.0, 1.0]]


# https://github.com/scipy/scipy/issues/8778#issuecomment-787603693
# error case (corrupt indptr)
def test_csr_tocsc_corrupt_indptr():
    with pytest.raises(
            ValueError,
            match="assignment destination is read-only"):
        data = [1.0, 1.0]
        indices = [0, 1]
        indptr = [0, 2]
        shape = (1, 2)
        m = csr_matrix((data, indices, indptr), shape=shape)
        m.indptr[1] = 10  # corrupt indptr
        # tocsc() does not validate the nnz value (the last item in the indptr
        # vector) before proceding
        m.tocsc()
        # np.sum(m)  # or other external function that we cannot check


# https://github.com/scipy/scipy/issues/8778#issuecomment-787603693
# error case (ok indptr)
def test_csr_tocsc_ok_indptr():
    data = [1.0, 1.0]
    indices = [0, 1]
    indptr = [0, 2]
    shape = (1, 2)
    m = csr_matrix((data, indices, indptr), shape=shape)
    # <no indptr corruption here>
    # tocsc() does not validate the nnz value (the last item in the indptr
    # vector) before proceding
    m.tocsc()


# https://github.com/scipy/scipy/issues/12131
# error case (corrupted indices: toarray)
def test_csr_toarray_corrupt_indices():
    with pytest.raises(
            AttributeError,
            match="'list' object has no attribute 'dtype'"):
        a = csr_matrix(np.arange(16).reshape((4, 4)))
        a.indices = [0]  # corrupt indices
        a.toarray()


# https://github.com/scipy/scipy/issues/12131
# error case (corrupted indices: sum)
def test_csr_sum_corrupt_indices():
    with pytest.raises(
            AttributeError,
            match="'list' object has no attribute 'dtype'"):
        a = csr_matrix(np.arange(16).reshape((4, 4)))
        a.indices = [0]  # corrupt indices
        np.sum(a)


# https://github.com/scipy/scipy/issues/12131
# error case (corrupted indices: at)
def test_csr_at_corrupt_indices():
    with pytest.raises(
            AttributeError,
            match="'list' object has no attribute 'dtype'"):
        a = csr_matrix(np.arange(16).reshape((4, 4)))
        b = csr_matrix(np.eye(4))
        a.indices = [0]  # corrupt indices
        a @ b
