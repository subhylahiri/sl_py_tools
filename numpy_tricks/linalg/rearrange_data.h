/* -*- Mode: C -*- */
/* Common code for creating GUFuncs with BLAS/Lapack
*/
#ifndef GUF_REARRANGE
#define GUF_REARRANGE
/*
*****************************************************************************
**                            Includes                                     **
*****************************************************************************
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "gufunc_common_f.h"
/*
*****************************************************************************
**                            Factories                                    **
*****************************************************************************
*/
#define DECLARE_FUNC_LINEARIZE(NAME, ...)                                                                     \
    void *                                                                                         \
    linearize_FLOAT_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);  \
    void *                                                                                         \
    linearize_DOUBLE_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void *                                                                                         \
    linearize_CFLOAT_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void *                                                                                         \
    linearize_CDOUBLE_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);

#define DECLARE_FUNC_DELINEARIZE(NAME, ...)                                                                     \
    void *                                                                                           \
    delinearize_FLOAT_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);  \
    void *                                                                                           \
    delinearize_DOUBLE_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void *                                                                                           \
    delinearize_CFLOAT_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void *                                                                                           \
    delinearize_CDOUBLE_## NAME(void *dst_in, const void *src_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);

#define DECLARE_FUNC_FILL(NAME, TYPE, ...)                                              \
    void                                                                     \
    NAME ##_FLOAT_## TYPE(void *dst_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);  \
    void                                                                     \
    NAME ##_DOUBLE_## TYPE(void *dst_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void                                                                     \
    NAME ##_CFLOAT_## TYPE(void *dst_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data); \
    void                                                                     \
    NAME ##_CDOUBLE_## TYPE(void *dst_in, const LINEARIZE_##__VA_ARGS__## DATA_t* data);
/*
*****************************************************************************
**                           Declarations                                  **
*****************************************************************************
*/
DECLARE_FUNC_LINEARIZE(matrix)
DECLARE_FUNC_DELINEARIZE(matrix)
DECLARE_FUNC_DELINEARIZE(triu)
DECLARE_FUNC_DELINEARIZE(tril)
DECLARE_FUNC_FILL(nan, matrix)
DECLARE_FUNC_FILL(zero, matrix)
DECLARE_FUNC_FILL(eye, matrix)
DECLARE_FUNC_LINEARIZE(vec, V)
DECLARE_FUNC_DELINEARIZE(vec, V)
DECLARE_FUNC_FILL(nan, vec, V)

void *
linearize_INT_vec(void *dst_in, const void *src_in, const LINEARIZE_VDATA_t* data);
void *
delinearize_INT_vec(void *dst_in, const void *src_in, const LINEARIZE_VDATA_t* data);

fortran_int
FLOAT_real_int(fortran_real val);
fortran_int
DOUBLE_real_int(fortran_doublereal val);
fortran_int
CFLOAT_real_int(fortran_complex val);
fortran_int
CDOUBLE_real_int(fortran_doublecomplex val);
/*
*****************************************************************************
*****************************************************************************
*/
#endif
