// Cleanup macro for accumulate_gene_norm_lm
#define CLEANUP_ACCUMULATE_GENE_NORM_LM()                                      \
    do {                                                                       \
        if (arr_vf_out) {                                                      \
            PyArray_ResolveWritebackIfCopy(arr_vf_out);                        \
            Py_DECREF(arr_vf_out);                                             \
            arr_vf_out = NULL;                                                 \
        }                                                                      \
        if (arr_selected) {                                                    \
            PyArray_ResolveWritebackIfCopy(arr_selected);                      \
            Py_DECREF(arr_selected);                                           \
            arr_selected = NULL;                                               \
        }                                                                      \
        Py_XDECREF(arr_vf_norm);                                               \
        arr_vf_norm = NULL;                                                    \
        Py_XDECREF(arr_lm_idx);                                                \
        arr_lm_idx = NULL;                                                     \
        Py_XDECREF(arr_lx);                                                    \
        arr_lx = NULL;                                                         \
        Py_XDECREF(arr_ly);                                                    \
        arr_ly = NULL;                                                         \
        Py_XDECREF(arr_lz);                                                    \
        arr_lz = NULL;                                                         \
    } while (0)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>
#include <queue>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
#endif

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#ifdef __AVX2__
  #include <immintrin.h> // AVX intrinsics
#endif

#include <Python.h>
#include "numpy/npy_math.h"
#include "numpy/arrayobject.h"

#define I2D(X, Y, YL) ((X) * (YL) + (Y))
#define I3D(X, Y, Z, YL, ZL) (((X) * (YL) * (ZL)) + ((Y) * (ZL)) + (Z))

struct pos2d {
    long x;
    long y;
};

struct pos3d {
    long x;
    long y;
    long z;
};

namespace std {
    template <>
    struct hash<pos3d> {
        std::size_t operator()(const pos3d &p) const {
            const std::size_t prime1 = 73856093;
            const std::size_t prime2 = 19349663;
            const std::size_t prime3 = 83492791;
            
            std::size_t hx = std::hash<long>()(p.x);
            std::size_t hy = std::hash<long>()(p.y);
            std::size_t hz = std::hash<long>()(p.z);
            
            return hx * prime1 ^ hy * prime2 ^ hz * prime3;
        }
    };
}

bool operator==(const pos3d& a, const pos3d& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static double gauss_kernel(double x, double y, double z) {
    return exp(-0.5 * (x*x + y*y + z*z));
}

void kde(std::unordered_map<pos3d, double> &arr, double *xx, double *yy, double *zz, int *shape, int npts, double bandwidth, double prune_coeff, int ncores) {
    int maxdist;
    if (prune_coeff > 0) {
        maxdist = static_cast<int>(bandwidth * prune_coeff);
    } else {
        maxdist = -1;
    }

    #pragma omp parallel num_threads(ncores)
    {
        std::unordered_map<pos3d, double> local_map;

        #pragma omp for
        for (int i = 0; i < npts; i++) {
            int x0 = static_cast<int>(xx[i]);
            int y0 = static_cast<int>(yy[i]);
            int z0 = static_cast<int>(zz[i]);

            int xs = (maxdist > 0) ? std::max(0, x0 - maxdist) : 0;
            int xe = (maxdist > 0) ? std::min(shape[0], x0 + maxdist + 1) : shape[0];
            int ys = (maxdist > 0) ? std::max(0, y0 - maxdist) : 0;
            int ye = (maxdist > 0) ? std::min(shape[1], y0 + maxdist + 1) : shape[1];
            int zs = (maxdist > 0) ? std::max(0, z0 - maxdist) : 0;
            int ze = (maxdist > 0) ? std::min(shape[2], z0 + maxdist + 1) : shape[2];

            for (int x = xs; x < xe; x++) {
                for (int y = ys; y < ye; y++) {
                    for (int z = zs; z < ze; z++) {
                        pos3d key{x, y, z};
                        double value = gauss_kernel((x - xx[i]) / bandwidth, (y - yy[i]) / bandwidth, (z - zz[i]) / bandwidth);
                        local_map[key] += value;
                    }
                }
            }
        }

        #pragma omp critical
        for (const auto& pair : local_map) {
            arr[pair.first] += pair.second;
        }
    }
}

static double __corr__(double *a, double *b, int ngene) {
    __m512d sum_a = _mm512_setzero_pd();
    __m512d sum_b = _mm512_setzero_pd();
    __m512d sum_aa = _mm512_setzero_pd();
    __m512d sum_bb = _mm512_setzero_pd();
    __m512d sum_ab_diff = _mm512_setzero_pd();

    int i;

    // Handle bulk of data with AVX-512
    for (i = 0; i <= ngene - 8; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);

        sum_a = _mm512_add_pd(sum_a, va);
        sum_b = _mm512_add_pd(sum_b, vb);
        
        sum_aa = _mm512_add_pd(sum_aa, _mm512_mul_pd(va, va));
        sum_bb = _mm512_add_pd(sum_bb, _mm512_mul_pd(vb, vb));
    }

    double arr_a[8], arr_b[8], arr_aa[8], arr_bb[8];
    _mm512_storeu_pd(arr_a, sum_a);
    _mm512_storeu_pd(arr_b, sum_b);
    _mm512_storeu_pd(arr_aa, sum_aa);
    _mm512_storeu_pd(arr_bb, sum_bb);

    double a_mean = 0, b_mean = 0, aa_mean = 0, bb_mean = 0;
    for(int j = 0; j < 8; j++) {
        a_mean += arr_a[j];
        b_mean += arr_b[j];
        aa_mean += arr_aa[j];
        bb_mean += arr_bb[j];
    }
    a_mean /= ngene;
    b_mean /= ngene;
    aa_mean /= ngene;
    bb_mean /= ngene;

    // Handle remainder
    for (; i < ngene; i++) {
        a_mean += a[i] / ngene;
        b_mean += b[i] / ngene;
        aa_mean += a[i] * a[i] / ngene;
        bb_mean += b[i] * b[i] / ngene;
    }

    double a_std = sqrt(aa_mean - a_mean * a_mean);
    double b_std = sqrt(bb_mean - b_mean * b_mean);
    double rtn = 0;

    if (a_std != 0 && b_std != 0) {
        for (i = 0; i <= ngene - 8; i += 8) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vb = _mm512_loadu_pd(&b[i]);

            va = _mm512_sub_pd(va, _mm512_set1_pd(a_mean));
            vb = _mm512_sub_pd(vb, _mm512_set1_pd(b_mean));

            sum_ab_diff = _mm512_add_pd(sum_ab_diff, _mm512_mul_pd(va, vb));
        }

        double arr_ab_diff[8];
        _mm512_storeu_pd(arr_ab_diff, sum_ab_diff);
        for(int j = 0; j < 8; j++) {
            rtn += arr_ab_diff[j];
        }

        for (; i < ngene; i++) {
            rtn += (a[i] - a_mean) * (b[i] - b_mean);
        }

        rtn /= (a_std * b_std * ngene);
    }

    return rtn;
}

static PyObject *check_avx512f(PyObject *self, PyObject *args) {
    long is_supported;
    int info[4];
    
    #ifdef _WIN32
    __cpuidex(info, 7, 0);
    #else
    __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
    #endif
    
    is_supported = (info[1] & (1 << 16)) != 0;  // Check if the 16th bit of EBX is set
    if (is_supported)
        return Py_True;
    else
        return Py_False;
}

static PyObject *calc_kde(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    PyArrayObject *arr3 = NULL;
    PyArrayObject *arr4 = NULL;
    int ncores = omp_get_max_threads();
    PyObject *rtn, *poslist, *xlist, *ylist, *zlist, *vlist;
    double *x, *y, *z;
    int *shape;
    double h, prune_coeff;
    int kernel = 0;
    unsigned int npts;
    int cnt;
    std::unordered_map<pos3d, double> oarr_map;
    std::unordered_map<pos3d, double>::iterator it;

    static const char *kwlist[] = { "h", "x", "y", "z", "shape", "prune_coeff", "kernel", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOOOd|ii", const_cast<char **>(kwlist), &h, &arg1, &arg2, &arg3, &arg4, &prune_coeff, &kernel, &ncores)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if ((arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if ((arr4 = (PyArrayObject*)PyArray_FROM_OTF(arg4, NPY_INT, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    
    if (PyArray_NDIM(arr1) != 1 || PyArray_NDIM(arr2) != 1 || PyArray_NDIM(arr3) != 1 || PyArray_NDIM(arr4) != 1)
    {
        goto fail;
    }

    npts = PyArray_DIMS(arr1)[0];

    x = (double *)PyArray_DATA(arr1);
    y = (double *)PyArray_DATA(arr2);
    z = (double *)PyArray_DATA(arr3);
    shape = (int *)PyArray_DATA(arr4);

    kde(oarr_map, x, y, z, shape, npts, h, prune_coeff, ncores);
    rtn = (PyObject *)PyTuple_New(2);
    poslist = (PyObject *)PyList_New(3);
    xlist = (PyObject *)PyList_New(oarr_map.size());
    ylist = (PyObject *)PyList_New(oarr_map.size());
    zlist = (PyObject *)PyList_New(oarr_map.size());
    vlist = (PyObject *)PyList_New(oarr_map.size());
    
    cnt = 0;
    for (const auto& pair : oarr_map) {
        PyList_SetItem(xlist, cnt, PyLong_FromLong((long)(pair.first.x)));
        PyList_SetItem(ylist, cnt, PyLong_FromLong((long)(pair.first.y)));
        PyList_SetItem(zlist, cnt, PyLong_FromLong((long)(pair.first.z)));
        PyList_SetItem(vlist, cnt, PyFloat_FromDouble(pair.second));
        cnt++;
    }
    PyList_SetItem(poslist, 0, xlist);
    PyList_SetItem(poslist, 1, ylist);
    PyList_SetItem(poslist, 2, zlist);
    PyTuple_SetItem(rtn, 0, poslist);
    PyTuple_SetItem(rtn, 1, vlist);
    
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(arr3);
    Py_DECREF(arr4);
    
    return (PyObject *) rtn;
    
fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    Py_XDECREF(arr3);
    Py_XDECREF(arr4);
    return NULL;
}

static PyObject *flood_fill(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject* filled_poslist = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    long nvec, nd, ngene = 0;
    long *pos, x, y, z, cnt;
    double r = 0.6, *vf;
    npy_intp *dimsp;
    int min_pixels = 10, max_pixels=2000;
    int i;
    bool *mask;

    static const char *kwlist[] = { "pos", "vf", "r", "min_pixels", "max_pixels", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dii", const_cast<char **>(kwlist), &arg1, &arg2, &r, &min_pixels, &max_pixels)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_LONG, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    nd = PyArray_NDIM(arr2);
    dimsp = PyArray_DIMS(arr2);
    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];
    ngene = dimsp[nd-1];
    filled_poslist = PyList_New(0);
    pos = (long *)PyArray_DATA(arr1);
    vf = (double *)PyArray_DATA(arr2);
    mask = (bool *) calloc(nvec, sizeof(bool));
    cnt = 0;
    if (nd == 3) {
        // 2D
        std::queue<pos2d> queue2d;
        queue2d.push(pos2d());
        queue2d.back().x = pos[0];
        queue2d.back().y = pos[1];
        while (queue2d.size() > 0) {
            x = queue2d.front().x;
            y = queue2d.front().y;
            PyObject *t = PyTuple_New(2);
            PyTuple_SetItem(t, 0, PyLong_FromLong(x));
            PyTuple_SetItem(t, 1, PyLong_FromLong(y));
            cnt += 1;
            if (cnt > max_pixels) 
                break;
            PyList_Append(filled_poslist, t);
            queue2d.pop();
            if (x < dimsp[0] - 1 && mask[I2D(x + 1, y, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x + 1, y, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x + 1, y, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x + 1;
                queue2d.back().y = y;
            }
            if (x > 1 && mask[I2D(x - 1, y, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x - 1, y, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x - 1, y, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x - 1;
                queue2d.back().y = y;
            }
            if (y < dimsp[1] - 1 && mask[I2D(x, y + 1, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x, y + 1, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x, y + 1, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x;
                queue2d.back().y = y + 1;
            }
            if (y > 1 && mask[I2D(x, y - 1, dimsp[1])] == false &&
                    __corr__(vf + (I2D(pos[0], pos[1], dimsp[1]) * ngene),
                             vf + (I2D(x, y - 1, dimsp[1]) * ngene), ngene) > r) {
                mask[I2D(x, y - 1, dimsp[1])] = true;
                queue2d.push(pos2d());
                queue2d.back().x = x;
                queue2d.back().y = y - 1;
            }
        }
    } else if (nd == 4) {
        // 3D
        std::queue<pos3d> queue3d;
        queue3d.push(pos3d());
        queue3d.back().x = pos[0];
        queue3d.back().y = pos[1];
        queue3d.back().z = pos[2];
        while (queue3d.size() > 0) {
            x = queue3d.front().x;
            y = queue3d.front().y;
            z = queue3d.front().z;
            PyObject *t = PyTuple_New(3);
            PyTuple_SetItem(t, 0, PyLong_FromLong(x));
            PyTuple_SetItem(t, 1, PyLong_FromLong(y));
            PyTuple_SetItem(t, 2, PyLong_FromLong(z));
            PyList_Append(filled_poslist, t);
            cnt += 1;
            if (cnt > max_pixels) 
                break;
            queue3d.pop();
            if (x < dimsp[0] - 1 && mask[I3D(x + 1, y, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x + 1, y, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x + 1, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x + 1;
                queue3d.back().y = y;
                queue3d.back().z = z;
            }
            if (x > 1 && mask[I3D(x - 1, y, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x - 1, y, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x - 1, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x - 1;
                queue3d.back().y = y;
                queue3d.back().z = z;
            }
            if (y < dimsp[1] - 1 && mask[I3D(x, y + 1, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y + 1, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y + 1, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y + 1;
                queue3d.back().z = z;
            }
            if (y > 1 && mask[I3D(x, y - 1, z, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y - 1, z, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y - 1, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y - 1;
                queue3d.back().z = z;
            }
            if (z < dimsp[2] - 1 && mask[I3D(x, y, z + 1, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y, z + 1, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y, z, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y;
                queue3d.back().z = z + 1;
            }
            if (z > 1 && mask[I3D(x, y, z - 1, dimsp[1], dimsp[2])] == false &&
                    __corr__(vf + I3D(pos[0], pos[1], pos[2], dimsp[1], dimsp[2]) * ngene,
                             vf + I3D(x, y, z - 1, dimsp[1], dimsp[2]) * ngene, ngene) > r) {
                mask[I3D(x, y, z - 1, dimsp[1], dimsp[2])] = true;
                queue3d.push(pos3d());
                queue3d.back().x = x;
                queue3d.back().y = y;
                queue3d.back().z = z - 1;
            }
        }
    }
    free((void*)mask);
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    if (cnt > max_pixels || cnt < min_pixels) 
        PyList_SetSlice(filled_poslist, 0, PyList_Size(filled_poslist), NULL);
    return (PyObject *) filled_poslist;
 
fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyObject *calc_corrmap(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *oarr = NULL;
    long i, x, y, z, dx, dy, dz;
    long nvec, nd, ngene = 0;
    double *vecs, *corrmap;
    npy_intp *dimsp;
    int ncores = omp_get_max_threads();
    int csize = 1;
    double *tmpvec;

    static const char *kwlist[] = { "vf", "ncores", "size", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", const_cast<char **>(kwlist), &arg1, &ncores, &csize)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;   
    nd = PyArray_NDIM(arr1);
    if (nd != 3 && nd != 4) goto fail; // only 2D or 3D array is expected
    dimsp = PyArray_DIMS(arr1);
    oarr = (PyArrayObject*)PyArray_ZEROS(nd - 1, dimsp, NPY_DOUBLE, NPY_CORDER);
    ngene = dimsp[nd-1];
    corrmap = (double *)PyArray_DATA(oarr);
    vecs = (double *)PyArray_DATA(arr1);
    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];
    
    // initialize corrmap with NANs
    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++)
        corrmap[i] = NPY_NAN;

    if (nd == 3) {
        // 2D
        #pragma omp parallel num_threads(ncores) private(tmpvec)
        {
            tmpvec = (double *)calloc(ngene, sizeof(double)); // zero initialized
            #pragma omp for collapse(2)
            for (x=csize; x<dimsp[0]-csize; x++) {
                for (y=csize; y<dimsp[1]-csize; y++) {
                    for (i=0; i<ngene; i++)
                        tmpvec[i] = 0;
                    for (dx=-csize; dx<csize+1; dx++) {
                        for (dy=-csize; dy<csize+1; dy++) {
                            if (dx == 0 && dy == 0) continue;
                            for (i=0; i<ngene; i++)
                                tmpvec[i] += (vecs + I2D(x+dx, y+dy, dimsp[1])*ngene)[i];
                        }
                    }
                    // tmpvec[i] /= (csize * 2 + 1) * (csize * 2 + 1) - 1;
                    corrmap[I2D(x, y, dimsp[1])] = __corr__(vecs + I2D(x, y, dimsp[1])*ngene, tmpvec, ngene);
                }
            }
            free((void*)tmpvec);
        }
    } else {
        // 3D (nd == 4)
        #pragma omp parallel num_threads(ncores) private(tmpvec)
        {
            tmpvec = (double *)calloc(ngene, sizeof(double));
            #pragma omp for collapse(3)
            for (x=csize; x<dimsp[0]-csize; x++) {
                for (y=csize; y<dimsp[1]-csize; y++) {
                    for (z=csize; z<dimsp[2]-csize; z++) {
                        for (i=0; i<ngene; i++)
                            tmpvec[i] = 0;
                        for (dx=-csize; dx<csize+1; dx++) {
                            for (dy=-csize; dy<csize+1; dy++) {
                                for (dz=-csize; dz<csize+1; dz++) {
                                    if (dx == 0 && dy == 0 && dz == 0) continue;
                                    for (i=0; i<ngene; i++)
                                        tmpvec[i] += (vecs + I3D(x+dx, y+dy, z+dz, dimsp[1], dimsp[2])*ngene)[i];
                                }
                            }
                        }
                        //for (i=0; i<ngene; i++)
                        //    tmpvec[i] /= (csize * 2 + 1) * (csize * 2 + 1) * (csize * 2 + 1) - 1;
                        corrmap[I3D(x, y, z, dimsp[1], dimsp[2])] =
                            __corr__(vecs + I3D(x, y, z, dimsp[1], dimsp[2])*ngene, tmpvec, ngene);
                    }
                }
            }
            free((void*)tmpvec);
        }
    }
    Py_DECREF(arr1);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    return NULL;
}

static PyObject *calc_corrmap_2(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *oarr = NULL;
    long i, k, x, y, z, dx, dy, dz;
    long nvec, nd, ngene = 0;
    double *vecs, *corrmap;
    npy_intp *dimsp;
    npy_intp dimsp2[4];
    int ncores = omp_get_max_threads();
    int csize = 1;

    static const char *kwlist[] = { "vf", "ncores", "size", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", const_cast<char **>(kwlist), &arg1, &ncores, &csize)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;   
    nd = PyArray_NDIM(arr1);
    if (nd != 3 && nd != 4) goto fail; // only 2D or 3D array is expected
    dimsp = PyArray_DIMS(arr1);
    for (i=0; i<nd-1; i++)
        dimsp2[i] = dimsp[i];
    dimsp2[nd-1] = (csize * 2 + 1) * (csize * 2 + 1) - 1;
    oarr = (PyArrayObject*)PyArray_ZEROS(nd, dimsp2, NPY_DOUBLE, NPY_CORDER);
    ngene = dimsp[nd-1];
    corrmap = (double *)PyArray_DATA(oarr);
    vecs = (double *)PyArray_DATA(arr1);
    nvec = 1;
    for (i=0; i<nd; i++)
        nvec *= dimsp2[i];

    // initialize corrmap with NANs
    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++)
        corrmap[i] = NPY_NAN;

    if (nd == 3) {
        // 2D
        #pragma omp parallel for collapse(2) private(k, dx, dy)
        for (x=csize; x<dimsp[0]-csize; x++) {
            for (y=csize; y<dimsp[1]-csize; y++) {
                k = 0;
                for (dx=-csize; dx<csize+1; dx++) {
                    for (dy=-csize; dy<csize+1; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        corrmap[I2D(x, y, dimsp2[1])*dimsp2[2] + (k++)] = 
                            __corr__(vecs + I2D(x, y, dimsp[1])*ngene,
                                     vecs + I2D(x+dx, y+dy, dimsp[1])*ngene, ngene);
                    }
                }
            }
        }
    } else {
        // 3D
        #pragma omp parallel for collapse(3) private(k, dx, dy, dz)
        for (x=csize; x<dimsp[0]-csize; x++) {
            for (y=csize; y<dimsp[1]-csize; y++) {
                for (z=csize; z<dimsp[2]-csize; z++) {
                    k = 0;
                    for (dx=-csize; dx<csize+1; dx++) {
                        for (dy=-csize; dy<csize+1; dy++) {
                            for (dz=-csize; dz<csize+1; dz++) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                corrmap[I3D(x, y, z, dimsp2[1], dimsp2[2])*dimsp2[3] + (k++)] =
                                    __corr__(vecs + I3D(x, y, z, dimsp[1], dimsp[2])*ngene, vecs + I3D(x+dx, y+dy, z+dz, dimsp[1], dimsp[2])*ngene, ngene);
                            }
                        }
                    }
                }
            }
        }
    }
    Py_DECREF(arr1);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    return NULL;
}

static PyObject *calc_ctmap(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    PyArrayObject *oarr = NULL;
    long nvec, nd, ngene = 0;
    double *cent, *vecs, *scores;
    npy_intp *dimsp;
    int ncores = omp_get_max_threads();
    int i;

    static const char *kwlist[] = { "vec", "vf", "ncores", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", const_cast<char **>(kwlist), &arg1, &arg2, &ncores)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    nd = PyArray_NDIM(arr2);
    if((ngene = *PyArray_DIMS(arr1)) != PyArray_DIMS(arr2)[nd-1]) goto fail;

    dimsp = PyArray_DIMS(arr2);
    oarr = (PyArrayObject*)PyArray_ZEROS(nd - 1, dimsp, NPY_DOUBLE, NPY_CORDER);

    nvec = 1;
    for (i=0; i<nd-1; i++)
        nvec *= dimsp[i];

    scores = (double *)PyArray_DATA(oarr);
    cent = (double *)PyArray_DATA(arr1);
    vecs = (double *)PyArray_DATA(arr2);

    #pragma omp parallel for num_threads(ncores)
    for (i=0; i<nvec; i++) {
        scores[i] = __corr__(cent, vecs + (i*ngene), ngene);
    }

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    return (PyObject *) oarr;
 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyObject *corr(PyObject *self, PyObject *args) {
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyArrayObject *arr1 = NULL;
    PyArrayObject *arr2 = NULL;
    long ngene = 0;
    double *a;
    double *b;
    double rtn = 0;

    if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;
    if ((arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) return NULL;
    if ((arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) goto fail;
    if (PyArray_NDIM(arr1) != 1) goto fail;
    if (PyArray_NDIM(arr2) != 1) goto fail;
    if((ngene = *PyArray_DIMS(arr1)) != *PyArray_DIMS(arr2)) goto fail;

    a = (double *)PyArray_DATA(arr1);
    b = (double *)PyArray_DATA(arr2);

    rtn = __corr__(a, b, ngene);

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    return PyFloat_FromDouble(rtn);
 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyObject *accumulate_gene(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg_vf = NULL;
    PyObject *arg_Xx = NULL;
    PyObject *arg_Xy = NULL;
    PyObject *arg_indptr = NULL;
    PyObject *arg_indices = NULL;
    PyObject *arg_dists = NULL;
    PyObject *arg_loc_z = NULL;
    
    PyArrayObject *arr_vf = NULL;
    PyArrayObject *arr_Xx = NULL;
    PyArrayObject *arr_Xy = NULL;
    PyArrayObject *arr_indptr = NULL;
    PyArrayObject *arr_indices = NULL;
    PyArrayObject *arr_dists = NULL;
    PyArrayObject *arr_loc_z = NULL;
    
    double sampling_distance, bandwidth, sigma;
    int gidx, dim;
    int ncores = omp_get_max_threads();

    static const char *kwlist[] = { "vf_temp", "Xx", "Xy", "indptr", "indices", "dists", 
                                   "loc_z", "sampling_distance", "bandwidth", "sigma", 
                                   "gidx", "dim", "ncores", NULL };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOdddii|i", const_cast<char **>(kwlist), 
                                    &arg_vf, &arg_Xx, &arg_Xy, &arg_indptr, &arg_indices, 
                                    &arg_dists, &arg_loc_z, &sampling_distance, &bandwidth, 
                                    &sigma, &gidx, &dim, &ncores)) return NULL;

    if ((arr_vf = (PyArrayObject*)PyArray_FROM_OTF(arg_vf, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY)) == NULL) return NULL;
    if ((arr_Xx = (PyArrayObject*)PyArray_FROM_OTF(arg_Xx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        return NULL;
    }
    if ((arr_Xy = (PyArrayObject*)PyArray_FROM_OTF(arg_Xy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        return NULL;
    }
    if ((arr_indptr = (PyArrayObject*)PyArray_FROM_OTF(arg_indptr, NPY_LONGLONG, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        return NULL;
    }
    if ((arr_indices = (PyArrayObject*)PyArray_FROM_OTF(arg_indices, NPY_LONGLONG, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        Py_XDECREF(arr_indptr);
        return NULL;
    }
    if ((arr_dists = (PyArrayObject*)PyArray_FROM_OTF(arg_dists, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        Py_XDECREF(arr_indptr);
        Py_XDECREF(arr_indices);
        return NULL;
    }
    if ((arr_loc_z = (PyArrayObject*)PyArray_FROM_OTF(arg_loc_z, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        Py_XDECREF(arr_indptr);
        Py_XDECREF(arr_indices);
        Py_XDECREF(arr_dists);
        return NULL;
    }

    if (PyArray_NDIM(arr_vf) != 4) {
        PyErr_SetString(PyExc_ValueError, "vf_temp must be 4D (X,Y,Z,G)");
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        Py_XDECREF(arr_indptr);
        Py_XDECREF(arr_indices);
        Py_XDECREF(arr_dists);
        Py_XDECREF(arr_loc_z);
        return NULL;
    }

    const int X = static_cast<int>(PyArray_DIM(arr_vf, 0));
    const int Y = static_cast<int>(PyArray_DIM(arr_vf, 1));
    const int Z = static_cast<int>(PyArray_DIM(arr_vf, 2));
    const int G = static_cast<int>(PyArray_DIM(arr_vf, 3));
    
    if (gidx < 0 || gidx >= G) {
        PyErr_SetString(PyExc_ValueError, "gidx out of range");
        Py_XDECREF(arr_vf);
        Py_XDECREF(arr_Xx);
        Py_XDECREF(arr_Xy);
        Py_XDECREF(arr_indptr);
        Py_XDECREF(arr_indices);
        Py_XDECREF(arr_dists);
        Py_XDECREF(arr_loc_z);
        return NULL;
    }

    const double* px = static_cast<double*>(PyArray_DATA(arr_Xx));
    const double* py = static_cast<double*>(PyArray_DATA(arr_Xy));
    const long long* p_indptr = static_cast<long long*>(PyArray_DATA(arr_indptr));
    const long long* p_indices = static_cast<long long*>(PyArray_DATA(arr_indices));
    const double* p_dists = static_cast<double*>(PyArray_DATA(arr_dists));
    const double* p_locz = static_cast<double*>(PyArray_DATA(arr_loc_z));

    const long long n_spot = PyArray_DIM(arr_indptr, 0) - 1;
    
    // 정규화 상수: (2*pi*h^2)^(d/2)
    const double norm = pow(2.0 * M_PI * (bandwidth * bandwidth), 0.5 * static_cast<double>(dim));

    // C-order strides
    float* base = static_cast<float*>(PyArray_DATA(arr_vf));
    const long long sX = (long long)Y * (long long)Z * (long long)G;
    const long long sY = (long long)Z * (long long)G;
    const long long sZ = (long long)G;
    const long long sG = 1;

    // 병렬 루프
    #pragma omp parallel for schedule(static) num_threads(ncores)
    for (long long i = 0; i < n_spot; ++i) {
        // spot i → z 인덱스
        int zi = static_cast<int>(floor(p_locz[i] / sampling_distance));
        if (zi < 0 || zi >= Z) continue;

        const long long start = p_indptr[i];
        const long long end = p_indptr[i + 1];

        for (long long p = start; p < end; ++p) {
            const long long idx = p_indices[p];  // grid index (row in X_)
            const double d = p_dists[p];

            // grid 좌표 → voxel 인덱스
            int xi = static_cast<int>(floor(px[idx] / sampling_distance));
            int yi = static_cast<int>(floor(py[idx] / sampling_distance));
            if (xi < 0 || xi >= X || yi < 0 || yi >= Y) continue;

            // 가우시안 값
            const double w = exp(-0.5 * (d / sigma) * (d / sigma)) / norm;
            const float wf = static_cast<float>(w);

            // 선형 오프셋
            const long long off = (long long)xi * sX + (long long)yi * sY + (long long)zi * sZ + (long long)gidx * sG;

            // 원자적 누적
            #if defined(_OPENMP)
              #pragma omp atomic
            #endif
            base[off] += wf;
        }
    }

    Py_DECREF(arr_vf);
    Py_DECREF(arr_Xx);
    Py_DECREF(arr_Xy);
    Py_DECREF(arr_indptr);
    Py_DECREF(arr_indices);
    Py_DECREF(arr_dists);
    Py_DECREF(arr_loc_z);

    Py_RETURN_NONE;
}

// static PyObject *accumulate_gene_vector(PyObject *self, PyObject *args, PyObject *kwargs) {
//     PyObject *arg_vf  = NULL;
//     PyObject *arg_lx  = NULL;
//     PyObject *arg_ly  = NULL;
//     PyObject *arg_lz  = NULL;

//     double sampling_distance, bandwidth, prune_coefficient;
//     int gidx, dim;
//     int ncores = omp_get_max_threads();

//     static const char *kwlist[] = {
//         "vf_temp", "loc_x", "loc_y", "loc_z",
//         "sampling_distance", "bandwidth", "prune_coefficient",
//         "gidx", "dim", "ncores", NULL
//     };

//     // 포맷 문자열 수정: OOOO ddd i i | i  → 공백 없이 "OOOOdddii|i"
//     if (!PyArg_ParseTupleAndKeywords(
//             args, kwargs, "OOOOdddii|i",
//             (char**)kwlist,
//             &arg_vf, &arg_lx, &arg_ly, &arg_lz,
//             &sampling_distance, &bandwidth, &prune_coefficient,
//             &gidx, &dim, &ncores))
//     {
//         return NULL;
//     }

//     // NumPy 배열 래핑
//     PyArrayObject *arr_vf = (PyArrayObject*)PyArray_FROM_OTF(arg_vf, NPY_FLOAT,  NPY_ARRAY_INOUT_ARRAY);
//     PyArrayObject *arr_lx = (PyArrayObject*)PyArray_FROM_OTF(arg_lx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     PyArrayObject *arr_ly = (PyArrayObject*)PyArray_FROM_OTF(arg_ly, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     PyArrayObject *arr_lz = (PyArrayObject*)PyArray_FROM_OTF(arg_lz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     if (!arr_vf || !arr_lx || !arr_ly || !arr_lz) {
//         Py_XDECREF(arr_vf); Py_XDECREF(arr_lx); Py_XDECREF(arr_ly); Py_XDECREF(arr_lz);
//         return NULL;
//     }

//     // vf_temp 모양 검사
//     if (PyArray_NDIM(arr_vf) != 4) {
//         PyErr_SetString(PyExc_ValueError, "vf_temp must be 4D (X,Y,Z,G)");
//         Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
//         PyArray_ResolveWritebackIfCopy(arr_vf);
//         Py_DECREF(arr_vf);
//         return NULL;
//     }

//     const npy_intp X = PyArray_DIM(arr_vf, 0);
//     const npy_intp Y = PyArray_DIM(arr_vf, 1);
//     const npy_intp Z = PyArray_DIM(arr_vf, 2);
//     const npy_intp G = PyArray_DIM(arr_vf, 3);

//     if (gidx < 0 || gidx >= (int)G) {
//         PyErr_SetString(PyExc_ValueError, "gidx out of range");
//         Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
//         PyArray_ResolveWritebackIfCopy(arr_vf);
//         Py_DECREF(arr_vf);
//         return NULL;
//     }

//     // 좌표 길이 동일성 체크
//     const npy_intp n_locs = PyArray_SIZE(arr_lx);
//     if (PyArray_SIZE(arr_ly) != n_locs || PyArray_SIZE(arr_lz) != n_locs) {
//         PyErr_SetString(PyExc_ValueError, "loc_x, loc_y, loc_z must have the same length");
//         Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
//         PyArray_ResolveWritebackIfCopy(arr_vf);
//         Py_DECREF(arr_vf);
//         return NULL;
//     }

//     // dim 검사
//     if (dim != 2 && dim != 3) {
//         PyErr_SetString(PyExc_ValueError, "dim must be 2 or 3");
//         Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
//         PyArray_ResolveWritebackIfCopy(arr_vf);
//         Py_DECREF(arr_vf);
//         return NULL;
//     }

//     const double *lx = (const double*)PyArray_DATA(arr_lx);
//     const double *ly = (const double*)PyArray_DATA(arr_ly);
//     const double *lz = (const double*)PyArray_DATA(arr_lz);

//     // 정규화 계수: (2*pi*h^2)^(d/2)
//     const double dimd     = (double)dim;
//     const double norm     = pow(2.0 * M_PI * bandwidth * bandwidth, 0.5 * dimd);  // h in world units
//     const double inv_norm = 1.0 / norm;

//     // 반경: effective_radius = int(bandwidth * prune_coefficient) [world] → pixel
//     const double Rw = bandwidth * prune_coefficient;                       // world
//     const npy_intp Rp = (npy_intp)floor(Rw / sampling_distance + 1e-12);   // pixel half-width

//     // 가우시안: exp(-r^2/(2*h^2))
//     const double two_h2 = 2.0 * bandwidth * bandwidth;

//     float *base = (float*)PyArray_DATA(arr_vf);

//     // 선형 오프셋 (C-order: ((x*Y)+y)*Z+z)*G + g)
//     const long long sX = (long long)Y * (long long)Z * (long long)G;
//     const long long sY = (long long)Z * (long long)G;
//     const long long sZ = (long long)G;
//     const long long sG = 1;

//     // 스레드 수 설정(선택)
// #ifdef _OPENMP
//     if (ncores < 1) ncores = 1;
//     if (ncores > omp_get_max_threads()) ncores = omp_get_max_threads();
// #endif

//     // 좌표 병렬 루프
//     #pragma omp parallel for schedule(dynamic, 16) num_threads(ncores)
//     for (npy_intp t = 0; t < n_locs; ++t) {
//         const double xw = lx[t];   // world x
//         const double yw = ly[t];   // world y
//         const double zw = (dim == 3 && Z > 1) ? lz[t] : 0.0; // depth==1이면 0으로 고정

//         // 중심 픽셀 인덱스(월드→픽셀)
//         const npy_intp xc = (npy_intp)floor(xw / sampling_distance);
//         const npy_intp yc = (npy_intp)floor(yw / sampling_distance);
//         const npy_intp zc = (npy_intp)floor(zw / sampling_distance);

//         // 윈도우 (클램프)
//         npy_intp x0 = xc - Rp, x1 = xc + Rp + 1;
//         npy_intp y0 = yc - Rp, y1 = yc + Rp + 1;
//         if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
//         if (x1 > X) x1 = X; if (y1 > Y) y1 = Y;

//         if (Z == 1) {
//             // ===== 2D 누적 =====
//             for (npy_intp xi = x0; xi < x1; ++xi) {
//                 const double xpw = (double)xi * sampling_distance; // 픽셀센터 world 좌표
//                 const double dxw = (xpw - xw);
//                 const double dxw2 = dxw * dxw;
//                 for (npy_intp yi = y0; yi < y1; ++yi) {
//                     const double ypw = (double)yi * sampling_distance;
//                     const double dyw  = (ypw - yw);
//                     const double r2   = dxw2 + dyw * dyw;
//                     if (r2 > Rw*Rw) continue; // 원형 마스크(선택): 사각창 내부에서 원 반경 제한

//                     const double w  = exp(-r2 / two_h2) * inv_norm;
//                     const float  wf = (float)w;

//                     const long long off =
//                         (long long)xi * sX + (long long)yi * sY + (long long)0 * sZ + (long long)gidx * sG;

//                     #if defined(_OPENMP)
//                       #pragma omp atomic
//                     #endif
//                     base[off] += wf;
//                 }
//             }
//         } else {
//             // ===== 3D 누적 =====
//             npy_intp z0 = zc - Rp, z1 = zc + Rp + 1;
//             if (z0 < 0) z0 = 0; if (z1 > Z) z1 = Z;

//             for (npy_intp xi = x0; xi < x1; ++xi) {
//                 const double xpw = (double)xi * sampling_distance;
//                 const double dxw = (xpw - xw);
//                 const double dxw2 = dxw * dxw;
//                 for (npy_intp yi = y0; yi < y1; ++yi) {
//                     const double ypw = (double)yi * sampling_distance;
//                     const double dyw = (ypw - yw);
//                     const double dxy2 = dxw2 + dyw * dyw;

//                     for (npy_intp zi = z0; zi < z1; ++zi) {
//                         const double zpw = (double)zi * sampling_distance;
//                         const double bzw = (zpw - zw);
//                         const double r2  = dxy2 + bzw * bzw;
//                         if (r2 > Rw*Rw) continue;

//                         const double w  = exp(-r2 / two_h2) * inv_norm;
//                         const float  wf = (float)w;

//                         const long long off =
//                             (long long)xi * sX + (long long)yi * sY + (long long)zi * sZ + (long long)gidx * sG;

//                         #if defined(_OPENMP)
//                           #pragma omp atomic
//                         #endif
//                         base[off] += wf;
//                     }
//                 }
//             }
//         }
//     }

//     // INOUT 배열 writeback
//     PyArray_ResolveWritebackIfCopy(arr_vf);
//     Py_DECREF(arr_vf);
//     Py_DECREF(arr_lx);
//     Py_DECREF(arr_ly);
//     Py_DECREF(arr_lz);

//     Py_RETURN_NONE;
// }


static PyObject *accumulate_gene_vector(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg_vf  = NULL;
    PyObject *arg_lx  = NULL;
    PyObject *arg_ly  = NULL;
    PyObject *arg_lz  = NULL;

    double sampling_distance, bandwidth, prune_coefficient;
    int gidx, dim;
    int ncores = omp_get_max_threads();
    /* chunk의 전역 시작 인덱스 (pixel 단위). 기본값 0 → 전체 vf와 동일 */
    int x0_chunk = 0, y0_chunk = 0, z0_chunk = 0;

    static const char *kwlist[] = {
        "vf_temp", "loc_x", "loc_y", "loc_z",
        "sampling_distance", "bandwidth", "prune_coefficient",
        "gidx", "dim", "ncores", "x0", "y0", "z0", NULL
    };

    /* 포맷: 필수 9개 + 옵션 4개 (ncores, x0, y0, z0) */
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOdddii|iiii",
            (char**)kwlist,
            &arg_vf, &arg_lx, &arg_ly, &arg_lz,
            &sampling_distance, &bandwidth, &prune_coefficient,
            &gidx, &dim,
            &ncores, &x0_chunk, &y0_chunk, &z0_chunk))
    {
        return NULL;
    }

    /* NumPy 배열 래핑 */
    PyArrayObject *arr_vf = (PyArrayObject*)PyArray_FROM_OTF(arg_vf, NPY_FLOAT,  NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *arr_lx = (PyArrayObject*)PyArray_FROM_OTF(arg_lx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_ly = (PyArrayObject*)PyArray_FROM_OTF(arg_ly, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lz = (PyArrayObject*)PyArray_FROM_OTF(arg_lz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr_vf || !arr_lx || !arr_ly || !arr_lz) {
        Py_XDECREF(arr_vf); Py_XDECREF(arr_lx); Py_XDECREF(arr_ly); Py_XDECREF(arr_lz);
        return NULL;
    }

    if (PyArray_NDIM(arr_vf) != 4) {
        PyErr_SetString(PyExc_ValueError, "vf_temp must be 4D (X,Y,Z,G)");
        Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
        PyArray_ResolveWritebackIfCopy(arr_vf);
        Py_DECREF(arr_vf);
        return NULL;
    }

    const npy_intp X = PyArray_DIM(arr_vf, 0);  /* chunk 크기 */
    const npy_intp Y = PyArray_DIM(arr_vf, 1);
    const npy_intp Z = PyArray_DIM(arr_vf, 2);
    const npy_intp G = PyArray_DIM(arr_vf, 3);

    if (gidx < 0 || gidx >= (int)G) {
        PyErr_SetString(PyExc_ValueError, "gidx out of range");
        Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
        PyArray_ResolveWritebackIfCopy(arr_vf);
        Py_DECREF(arr_vf);
        return NULL;
    }

    const npy_intp n_locs = PyArray_SIZE(arr_lx);
    if (PyArray_SIZE(arr_ly) != n_locs || PyArray_SIZE(arr_lz) != n_locs) {
        PyErr_SetString(PyExc_ValueError, "loc_x, loc_y, loc_z must have the same length");
        Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
        PyArray_ResolveWritebackIfCopy(arr_vf);
        Py_DECREF(arr_vf);
        return NULL;
    }

    if (dim != 2 && dim != 3) {
        PyErr_SetString(PyExc_ValueError, "dim must be 2 or 3");
        Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
        PyArray_ResolveWritebackIfCopy(arr_vf);
        Py_DECREF(arr_vf);
        return NULL;
    }

    const double *lx = (const double*)PyArray_DATA(arr_lx);
    const double *ly = (const double*)PyArray_DATA(arr_ly);
    const double *lz = (const double*)PyArray_DATA(arr_lz);

    const double dimd     = (double)dim;
    const double norm     = pow(2.0 * M_PI * bandwidth * bandwidth, 0.5 * dimd);
    const double inv_norm = 1.0 / norm;

    const double Rw = bandwidth * prune_coefficient;
    const npy_intp Rp = (npy_intp)floor(Rw / sampling_distance + 1e-12);
    const double two_h2 = 2.0 * bandwidth * bandwidth;

    float *base = (float*)PyArray_DATA(arr_vf);

    /* chunk의 전역 범위 (pixel index) */
    const npy_intp gx0 = (npy_intp)x0_chunk;
    const npy_intp gy0 = (npy_intp)y0_chunk;
    const npy_intp gz0 = (npy_intp)z0_chunk;
    const npy_intp gx1 = gx0 + X;   /* [gx0, gx1) */
    const npy_intp gy1 = gy0 + Y;
    const npy_intp gz1 = gz0 + Z;

    const long long sX = (long long)Y * (long long)Z * (long long)G;
    const long long sY = (long long)Z * (long long)G;
    const long long sZ = (long long)G;
    const long long sG = 1;

#ifdef _OPENMP
    if (ncores < 1) ncores = 1;
    if (ncores > omp_get_max_threads()) ncores = omp_get_max_threads();
#endif

    #pragma omp parallel for schedule(dynamic, 16) num_threads(ncores)
    for (npy_intp t = 0; t < n_locs; ++t) {
        const double xw = lx[t];
        const double yw = ly[t];
        const double zw = (dim == 3 && Z > 1) ? lz[t] : 0.0;

        /* 전역 픽셀 인덱스 (전체 그리드 기준) */
        const npy_intp xc_g = (npy_intp)floor(xw / sampling_distance);
        const npy_intp yc_g = (npy_intp)floor(yw / sampling_distance);
        const npy_intp zc_g = (npy_intp)floor(zw / sampling_distance);

        /* 전역 윈도우 */
        npy_intp x0g = xc_g - Rp, x1g = xc_g + Rp + 1;
        npy_intp y0g = yc_g - Rp, y1g = yc_g + Rp + 1;

        /* 이 chunk가 커버하는 전역 범위와의 교집합만 사용 */
        if (x0g < gx0) x0g = gx0;
        if (y0g < gy0) y0g = gy0;
        if (x1g > gx1) x1g = gx1;
        if (y1g > gy1) y1g = gy1;
        if (x0g >= x1g || y0g >= y1g) {
            continue;  /* 이 좌표에서 나오는 KDE가 이 chunk에는 기여 없음 */
        }

        if (Z == 1) {
            /* ===== 2D 누적 (z=0만 존재) ===== */
            for (npy_intp xg = x0g; xg < x1g; ++xg) {
                const double xpw = (double)xg * sampling_distance;
                const double dxw = (xpw - xw);
                const double dxw2 = dxw * dxw;

                const npy_intp xi = xg - gx0;  /* local index */

                for (npy_intp yg = y0g; yg < y1g; ++yg) {
                    const double ypw = (double)yg * sampling_distance;
                    const double dyw  = (ypw - yw);
                    const double r2   = dxw2 + dyw * dyw;
                    if (r2 > Rw * Rw) continue;

                    const double w  = exp(-r2 / two_h2) * inv_norm;
                    const float  wf = (float)w;

                    const npy_intp yi = yg - gy0;  /* local index */
                    const long long off =
                        (long long)xi * sX +
                        (long long)yi * sY +
                        (long long)0  * sZ +
                        (long long)gidx * sG;

                    #if defined(_OPENMP)
                      #pragma omp atomic
                    #endif
                    base[off] += wf;
                }
            }
        } else {
            /* ===== 3D 누적 ===== */
            npy_intp z0g = zc_g - Rp, z1g = zc_g + Rp + 1;
            if (z0g < gz0) z0g = gz0;
            if (z1g > gz1) z1g = gz1;
            if (z0g >= z1g) continue;

            for (npy_intp xg = x0g; xg < x1g; ++xg) {
                const double xpw = (double)xg * sampling_distance;
                const double dxw = (xpw - xw);
                const double dxw2 = dxw * dxw;
                const npy_intp xi = xg - gx0;

                for (npy_intp yg = y0g; yg < y1g; ++yg) {
                    const double ypw = (double)yg * sampling_distance;
                    const double dyw = (ypw - yw);
                    const double dxy2 = dxw2 + dyw * dyw;
                    const npy_intp yi = yg - gy0;

                    for (npy_intp zg = z0g; zg < z1g; ++zg) {
                        const double zpw = (double)zg * sampling_distance;
                        const double bzw = (zpw - zw);
                        const double r2  = dxy2 + bzw * bzw;
                        if (r2 > Rw * Rw) continue;

                        const double w  = exp(-r2 / two_h2) * inv_norm;
                        const float  wf = (float)w;

                        const npy_intp zi = zg - gz0;  /* local index */
                        const long long off =
                            (long long)xi * sX +
                            (long long)yi * sY +
                            (long long)zi * sZ +
                            (long long)gidx * sG;

                        #if defined(_OPENMP)
                          #pragma omp atomic
                        #endif
                        base[off] += wf;
                    }
                }
            }
        }
    }

    PyArray_ResolveWritebackIfCopy(arr_vf);
    Py_DECREF(arr_vf);
    Py_DECREF(arr_lx);
    Py_DECREF(arr_ly);
    Py_DECREF(arr_lz);

    Py_RETURN_NONE;
}


/* ===================================================================
 * kde_at_points  –  Compute KDE vectors at arbitrary query positions.
 *
 * This is the "gather" counterpart of accumulate_gene_vector (scatter).
 * For each query point, it finds all transcripts within the pruning
 * radius and accumulates Gaussian weights, producing a raw KDE vector
 * per query point.  The result is mathematically identical to reading
 * the value from a full grid computed by accumulate_gene_vector, but
 * only for the requested positions (sparse evaluation).
 *
 * Python signature:
 *   kde_at_points(
 *       query_x,            # float64 (N,)  query µm coords
 *       query_y,            # float64 (N,)
 *       query_z,            # float64 (N,)
 *       loc_x,              # float64 (M,)  transcript µm coords
 *       loc_y,              # float64 (M,)
 *       loc_z,              # float64 (M,)
 *       gene_indices,       # int32   (M,)  gene index per transcript
 *       n_genes,            # int
 *       sampling_distance,  # double
 *       bandwidth,          # double
 *       prune_coefficient,  # double
 *       dim,                # int  (2 or 3)
 *       ncores=auto,        # int  (optional)
 *   ) -> ndarray float32 (N, n_genes)
 *
 * The function bins transcripts into a grid (cell size = sampling_distance)
 * and for each query point only visits grid cells within Rp of the
 * query's own grid cell — exactly the same indexing logic as
 * accumulate_gene_vector.
 * =================================================================== */
static PyObject *kde_at_points(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *arg_qx = NULL, *arg_qy = NULL, *arg_qz = NULL;
    PyObject *arg_lx = NULL, *arg_ly = NULL, *arg_lz = NULL;
    PyObject *arg_gi = NULL;
    int n_genes, dim;
    double sampling_distance, bandwidth, prune_coefficient;
    int ncores = omp_get_max_threads();

    static const char *kwlist[] = {
        "query_x", "query_y", "query_z",
        "loc_x", "loc_y", "loc_z",
        "gene_indices", "n_genes",
        "sampling_distance", "bandwidth", "prune_coefficient",
        "dim", "ncores", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOidddi|i",
            (char**)kwlist,
            &arg_qx, &arg_qy, &arg_qz,
            &arg_lx, &arg_ly, &arg_lz,
            &arg_gi, &n_genes,
            &sampling_distance, &bandwidth, &prune_coefficient,
            &dim, &ncores))
    {
        return NULL;
    }

    /* --- NumPy array wrappers --- */
    PyArrayObject *arr_qx = (PyArrayObject*)PyArray_FROM_OTF(arg_qx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_qy = (PyArrayObject*)PyArray_FROM_OTF(arg_qy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_qz = (PyArrayObject*)PyArray_FROM_OTF(arg_qz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lx = (PyArrayObject*)PyArray_FROM_OTF(arg_lx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_ly = (PyArrayObject*)PyArray_FROM_OTF(arg_ly, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_lz = (PyArrayObject*)PyArray_FROM_OTF(arg_lz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arr_gi = (PyArrayObject*)PyArray_FROM_OTF(arg_gi, NPY_INT32,  NPY_ARRAY_IN_ARRAY);
    if (!arr_qx || !arr_qy || !arr_qz || !arr_lx || !arr_ly || !arr_lz || !arr_gi) {
        Py_XDECREF(arr_qx); Py_XDECREF(arr_qy); Py_XDECREF(arr_qz);
        Py_XDECREF(arr_lx); Py_XDECREF(arr_ly); Py_XDECREF(arr_lz);
        Py_XDECREF(arr_gi);
        return NULL;
    }

    const npy_intp N = PyArray_SIZE(arr_qx);  /* number of query points */
    const npy_intp M = PyArray_SIZE(arr_lx);  /* number of transcripts */

    if (PyArray_SIZE(arr_qy) != N || PyArray_SIZE(arr_qz) != N) {
        PyErr_SetString(PyExc_ValueError, "query_x/y/z must have the same length");
        goto fail;
    }
    if (PyArray_SIZE(arr_ly) != M || PyArray_SIZE(arr_lz) != M || PyArray_SIZE(arr_gi) != M) {
        PyErr_SetString(PyExc_ValueError, "loc_x/y/z and gene_indices must have the same length");
        goto fail;
    }
    if (dim != 2 && dim != 3) {
        PyErr_SetString(PyExc_ValueError, "dim must be 2 or 3");
        goto fail;
    }

    {
        const double *qx = (const double*)PyArray_DATA(arr_qx);
        const double *qy = (const double*)PyArray_DATA(arr_qy);
        const double *qz = (const double*)PyArray_DATA(arr_qz);
        const double *lx = (const double*)PyArray_DATA(arr_lx);
        const double *ly = (const double*)PyArray_DATA(arr_ly);
        const double *lz = (const double*)PyArray_DATA(arr_lz);
        const int    *gi = (const int*)PyArray_DATA(arr_gi);

        const double dimd     = (double)dim;
        const double norm     = pow(2.0 * M_PI * bandwidth * bandwidth, 0.5 * dimd);
        const double inv_norm = 1.0 / norm;
        const double Rw       = bandwidth * prune_coefficient;
        const double Rw2      = Rw * Rw;
        const npy_intp Rp     = (npy_intp)floor(Rw / sampling_distance + 1e-12);
        const double two_h2   = 2.0 * bandwidth * bandwidth;
        const double inv_sd   = 1.0 / sampling_distance;

        /* --- Allocate output array (N, n_genes), zero-initialized --- */
        npy_intp out_dims[2] = {N, (npy_intp)n_genes};
        PyArrayObject *arr_out = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_FLOAT, 0);
        if (!arr_out) { goto fail; }

        float *out = (float*)PyArray_DATA(arr_out);

        /* ---------------------------------------------------------------
         * Build a grid-cell index over transcripts.
         *
         * Each grid cell (cx, cy) holds a list of transcript indices.
         * Cell size = sampling_distance.  We use a hash map keyed by
         * (cx, cy) packed into a 64-bit integer.
         * --------------------------------------------------------------- */

        /* Compute grid cell for each transcript */
        std::vector<npy_intp> tcx(M), tcy(M);
        for (npy_intp t = 0; t < M; ++t) {
            tcx[t] = (npy_intp)floor(lx[t] * inv_sd);
            tcy[t] = (npy_intp)floor(ly[t] * inv_sd);
        }

        /* Pack (cx,cy) into int64 key. Shift cx by 32 bits. */
        auto pack_key = [](npy_intp cx, npy_intp cy) -> int64_t {
            return ((int64_t)cx << 32) ^ (int64_t)(uint32_t)cy;
        };

        /* Build hash map: grid_cell_key -> list of transcript indices */
        std::unordered_map<int64_t, std::vector<npy_intp>> grid;
        grid.reserve((size_t)(M / 4 + 1));
        for (npy_intp t = 0; t < M; ++t) {
            grid[pack_key(tcx[t], tcy[t])].push_back(t);
        }

        /* ---------------------------------------------------------------
         * For each query point, iterate over grid cells within Rp and
         * accumulate KDE contributions.
         * --------------------------------------------------------------- */
#ifdef _OPENMP
        if (ncores < 1) ncores = 1;
        if (ncores > omp_get_max_threads()) ncores = omp_get_max_threads();
#endif

        const bool is_2d = (dim == 2);

        #pragma omp parallel for schedule(dynamic, 64) num_threads(ncores)
        for (npy_intp i = 0; i < N; ++i) {
            const double qxw = qx[i];
            const double qyw = qy[i];
            const double qzw = is_2d ? 0.0 : qz[i];

            const npy_intp qcx = (npy_intp)floor(qxw * inv_sd);
            const npy_intp qcy = (npy_intp)floor(qyw * inv_sd);

            float *row = out + (npy_intp)i * (npy_intp)n_genes;

            /* Visit all grid cells within Rp (Chebyshev distance) */
            for (npy_intp cx = qcx - Rp; cx <= qcx + Rp; ++cx) {
                for (npy_intp cy = qcy - Rp; cy <= qcy + Rp; ++cy) {
                    auto it = grid.find(pack_key(cx, cy));
                    if (it == grid.end()) continue;

                    const std::vector<npy_intp> &indices = it->second;
                    for (npy_intp t : indices) {
                        const double dx = lx[t] - qxw;
                        const double dy = ly[t] - qyw;
                        double r2 = dx * dx + dy * dy;

                        if (!is_2d) {
                            const double dz = lz[t] - qzw;
                            r2 += dz * dz;
                        }

                        if (r2 > Rw2) continue;

                        const double w  = exp(-r2 / two_h2) * inv_norm;
                        const int    g  = gi[t];
                        if (g >= 0 && g < n_genes) {
                            row[g] += (float)w;
                        }
                    }
                }
            }
        }

        /* --- Cleanup and return --- */
        Py_DECREF(arr_qx); Py_DECREF(arr_qy); Py_DECREF(arr_qz);
        Py_DECREF(arr_lx); Py_DECREF(arr_ly); Py_DECREF(arr_lz);
        Py_DECREF(arr_gi);
        return (PyObject*)arr_out;
    }

fail:
    Py_XDECREF(arr_qx); Py_XDECREF(arr_qy); Py_XDECREF(arr_qz);
    Py_XDECREF(arr_lx); Py_XDECREF(arr_ly); Py_XDECREF(arr_lz);
    Py_XDECREF(arr_gi);
    return NULL;
}


// static PyObject *normalize_vf_sparse(PyObject *self, PyObject *args, PyObject *kwargs){
//     PyObject *vf_chunk_obj = NULL;
//     PyObject *vf_norm_obj  = NULL;
//     double size_after_norm;
//     int x0;
//     int ncores;

//     static const char *kwlist[] = {
//         "vf_chunk", "vf_norm",
//         "size_after_norm",
//         "x0", "ncores",
//         NULL
//     };

//     if (!PyArg_ParseTupleAndKeywords(
//             args, kwargs, "OOdii",
//             (char**)kwlist,
//             &vf_chunk_obj, &vf_norm_obj,
//             &size_after_norm,
//             &x0, &ncores
//         )) {
//         return NULL;
//     }

//     /* NumPy array로 변환 (float32, C-contiguous 보장) */
//     PyArrayObject *vf_chunk_arr = (PyArrayObject*)PyArray_FROMANY(
//         vf_chunk_obj, NPY_FLOAT32, 4, 4, NPY_ARRAY_C_CONTIGUOUS);
//     PyArrayObject *vf_norm_arr = (PyArrayObject*)PyArray_FROMANY(
//         vf_norm_obj, NPY_FLOAT32, 3, 3, NPY_ARRAY_C_CONTIGUOUS);

//     if (vf_chunk_arr == NULL || vf_norm_arr == NULL) {
//         Py_XDECREF(vf_chunk_arr);
//         Py_XDECREF(vf_norm_arr);
//         return NULL;
//     }

//     float *vf_chunk = (float*)PyArray_DATA(vf_chunk_arr);
//     float *vf_norm  = (float*)PyArray_DATA(vf_norm_arr);

//     npy_intp *shape_chunk = PyArray_SHAPE(vf_chunk_arr);
//     npy_intp *shape_norm  = PyArray_SHAPE(vf_norm_arr);

//     int cx = (int)shape_chunk[0];
//     int vy = (int)shape_chunk[1];
//     int vz = (int)shape_chunk[2];
//     int ng = (int)shape_chunk[3];

//     /* 간단한 shape 일치 체크 */
//     if (shape_norm[0] != cx || shape_norm[1] != vy || shape_norm[2] != vz) {
//         Py_DECREF(vf_chunk_arr);
//         Py_DECREF(vf_norm_arr);
//         PyErr_SetString(PyExc_ValueError, "vf_chunk and vf_norm shape mismatch");
//         return NULL;
//     }

//     long nvox = (long)cx * (long)vy * (long)vz;

//     /* 1차 패스: nnz 개수 세기 (vf_norm > 0 && vf_chunk != 0) */
//     long nnz = 0;

//     #pragma omp parallel for reduction(+:nnz) num_threads(ncores)
//     for (long vidx = 0; vidx < nvox; ++vidx) {
//         float norm = vf_norm[vidx];
//         if (norm <= 0.0f)
//             continue;

//         long base = vidx * (long)ng;
//         for (int g = 0; g < ng; ++g) {
//             float v = vf_chunk[base + g];
//             if (v != 0.0f) {
//                 nnz++;
//             }
//         }
//     }

//     /* nnz가 0이면 빈 배열 반환 */
//     npy_intp shp1[1] = { nnz };
//     PyArrayObject *x_arr    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, shp1, NPY_INT32));
//     PyArrayObject *y_arr    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, shp1, NPY_INT32));
//     PyArrayObject *z_arr    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, shp1, NPY_INT32));
//     PyArrayObject *g_arr    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, shp1, NPY_INT32));
//     PyArrayObject *data_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, shp1, NPY_FLOAT32));

//     int   *out_x = static_cast<int*>(PyArray_DATA(x_arr));
//     int   *out_y = static_cast<int*>(PyArray_DATA(y_arr));
//     int   *out_z = static_cast<int*>(PyArray_DATA(z_arr));
//     int   *out_g = static_cast<int*>(PyArray_DATA(g_arr));
//     float *out_v = static_cast<float*>(PyArray_DATA(data_arr));

//     long pos = 0;

//     /* 2차 패스: 실제 값 채우기 */
//     #pragma omp parallel for num_threads(ncores) schedule(static)
//     for (long vidx = 0; vidx < nvox; ++vidx) {
//         float norm = vf_norm[vidx];
//         if (norm <= 0.0f)
//             continue;

//         float scale = (float)(size_after_norm / (double)norm);

//         long tmp = vidx;
//         int iz = (int)(tmp % vz); tmp /= vz;
//         int iy = (int)(tmp % vy);
//         int ix = (int)(tmp / vy);

//         long base = vidx * (long)ng;

//         for (int g = 0; g < ng; ++g) {
//             float v0 = vf_chunk[base + g];
//             if (v0 == 0.0f)
//                 continue;

//             float v = v0 * scale;

//             long my_pos;
//             #pragma omp atomic capture
//             { my_pos = pos; pos++; }

//             out_x[my_pos] = ix + x0;
//             out_y[my_pos] = iy;
//             out_z[my_pos] = iz;
//             out_g[my_pos] = g;
//             out_v[my_pos] = v;
//         }
//     }

//     Py_DECREF(vf_chunk_arr);
//     Py_DECREF(vf_norm_arr);

//     /* (x, y, z, g, data) 튜플로 반환 */
//     return Py_BuildValue("OOOOO", x_arr, y_arr, z_arr, g_arr, data_arr);
// }

// /* 모듈 메서드 테이블 */
// static PyMethodDef ssam_norm_methods[] = {
//     {
//         "normalize_vf_sparse",
//         (PyCFunction)normalize_vf_sparse,
//         METH_VARARGS | METH_KEYWORDS,
//         "Normalize vf_chunk by vf_norm and return sparse (x,y,z,g,data)."
//     },
//     {NULL, NULL, 0, NULL}
// };

// /* 모듈 정의 */
// static struct PyModuleDef ssam_norm_module = {
//     PyModuleDef_HEAD_INIT,
//     "ssam_norm",
//     "Sparse normalization for SSAM",
//     -1,
//     ssam_norm_methods
// };

// /* 모듈 초기화 */
// PyMODINIT_FUNC
// PyInit_ssam_norm(void)
// {
//     import_array();
//     return PyModule_Create(&ssam_norm_module);
// }



// static PyObject* normalize_vf(
//     PyObject *self, PyObject *args, PyObject *kwargs)
// {
//     PyArrayObject *vf_chunk_obj = NULL;
//     PyArrayObject *vf_norm_obj  = NULL;

//     double size_after_norm;
//     int x0, ncores;

//     static const char *kwlist[] = {
//         "vf_chunk", "vf_norm",
//         "size_after_norm",
//         "x0", "ncores",
//         NULL
//     };

//     if (!PyArg_ParseTupleAndKeywords(
//             args, kwargs, "OOdii",
//             (char**)kwlist,
//             &vf_chunk_obj, &vf_norm_obj,
//             &size_after_norm,
//             &x0, &ncores
//         )) {
//         return NULL;
//     }

//     float *vf_chunk = (float*)PyArray_DATA(vf_chunk_obj);
//     float *vf_norm  = (float*)PyArray_DATA(vf_norm_obj);

//     npy_intp *shape = PyArray_SHAPE(vf_chunk_obj);
//     int cx = shape[0];
//     int vy = shape[1];
//     int vz = shape[2];
//     int ng = shape[3];

//     /* 1) nonzero count */
//     long total = (long)cx * vy * vz;

//     long count = 0;

//     #pragma omp parallel for reduction(+:count) num_threads(ncores)
//     for (long idx = 0; idx < total; idx++) {
//         if (vf_norm[idx] > 0.0f)
//             count++;
//     }

//     /* allocate output arrays */
//     npy_intp out_shape1[1] = { count };
//     PyArrayObject *x_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, out_shape1, NPY_INT32));
//     PyArrayObject *y_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, out_shape1, NPY_INT32));
//     PyArrayObject *z_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, out_shape1, NPY_INT32));

//     npy_intp out_shape2[2] = { count, ng };
//     PyArrayObject *data_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, out_shape2, NPY_FLOAT32));

//     int *out_x = static_cast<int*>(PyArray_DATA(x_arr));
//     int *out_y = static_cast<int*>(PyArray_DATA(y_arr));
//     int *out_z = static_cast<int*>(PyArray_DATA(z_arr));
//     float *out_data = static_cast<float*>(PyArray_DATA(data_arr));

//     /* 2) fill outputs */
//     long pos = 0;

//     #pragma omp parallel num_threads(ncores)
//     {
//         /* local buffers for thread-local writes */
//         long local_pos = 0;

//         /* thread-local lists (store indices only), then commit once */
//         int *lx = static_cast<int*>(malloc(count * sizeof(int)));
//         int *ly = static_cast<int*>(malloc(count * sizeof(int)));
//         int *lz = static_cast<int*>(malloc(count * sizeof(int)));
//         float *ldata = static_cast<float*>(malloc(count * ng * sizeof(float)));

//         #pragma omp for nowait
//         for (long idx = 0; idx < total; idx++) {

//             if (vf_norm[idx] <= 0.0f)
//                 continue;

//             /* unravel index */
//             long tmp = idx;
//             int iz = tmp % vz;   tmp /= vz;
//             int iy = tmp % vy;
//             int ix = tmp / vy;

//             float norm = vf_norm[idx];
//             float scale = (float)(size_after_norm / norm);

//             /* store coords */
//             lx[local_pos] = ix + x0;
//             ly[local_pos] = iy;
//             lz[local_pos] = iz;

//             /* extract vector */
//             float *src = vf_chunk + (long)idx * ng;
//             float *dst = ldata + (long)local_pos * ng;

//             /* L1 normalization only */
//             for (int g = 0; g < ng; g++)
//                 dst[g] = src[g] * scale;

//             local_pos++;
//         }

//         /* commit to global arrays */
//         long offset;

//         #pragma omp atomic capture
//         {
//             offset = pos;
//             pos += local_pos;
//         }

//         memcpy(out_x + offset, lx, local_pos * sizeof(int));
//         memcpy(out_y + offset, ly, local_pos * sizeof(int));
//         memcpy(out_z + offset, lz, local_pos * sizeof(int));
//         memcpy(out_data + (long)offset * ng, ldata, (long)local_pos * ng * sizeof(float));

//         free(lx);
//         free(ly);
//         free(lz);
//         free(ldata);
//     }

//     /* return (x, y, z, data) */
//     return Py_BuildValue("OOOO", x_arr, y_arr, z_arr, data_arr);
// }

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#include <vector>
#include <algorithm>
#include <limits>

static PyObject* normalize_vf(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *vf_chunk_in = NULL;
    PyObject *vf_norm_in  = NULL;

    double size_after_norm;
    int x0, ncores;

    static const char *kwlist[] = {
        "vf_chunk", "vf_norm",
        "size_after_norm",
        "x0", "ncores",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOdii",
            (char**)kwlist,
            &vf_chunk_in, &vf_norm_in,
            &size_after_norm, &x0, &ncores))
    {
        return NULL;
    }

    // --- 0) Enforce float32 + contiguous for safety (and DECREF later)
    PyArrayObject *vf_chunk_obj = (PyArrayObject*)PyArray_FROM_OTF(
        vf_chunk_in, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    if (!vf_chunk_obj) return NULL;

    PyArrayObject *vf_norm_obj = (PyArrayObject*)PyArray_FROM_OTF(
        vf_norm_in, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY
    );
    if (!vf_norm_obj) { Py_DECREF(vf_chunk_obj); return NULL; }

    // --- shape checks
    if (PyArray_NDIM(vf_chunk_obj) != 4) {
        Py_DECREF(vf_chunk_obj);
        Py_DECREF(vf_norm_obj);
        PyErr_SetString(PyExc_ValueError, "vf_chunk must be 4D (cx, vy, vz, ng)");
        return NULL;
    }
    if (PyArray_NDIM(vf_norm_obj) != 3 && PyArray_NDIM(vf_norm_obj) != 1) {
        Py_DECREF(vf_chunk_obj);
        Py_DECREF(vf_norm_obj);
        PyErr_SetString(PyExc_ValueError, "vf_norm must be 3D (cx, vy, vz) or flattened 1D");
        return NULL;
    }

    npy_intp *sh = PyArray_SHAPE(vf_chunk_obj);
    const npy_intp cx = sh[0];
    const npy_intp vy = sh[1];
    const npy_intp vz = sh[2];
    const npy_intp ng = sh[3];

    const npy_intp total = cx * vy * vz;

    // vf_norm length compatibility
    npy_intp norm_total = 0;
    if (PyArray_NDIM(vf_norm_obj) == 3) {
        npy_intp *nsh = PyArray_SHAPE(vf_norm_obj);
        if (nsh[0] != cx || nsh[1] != vy || nsh[2] != vz) {
            Py_DECREF(vf_chunk_obj);
            Py_DECREF(vf_norm_obj);
            PyErr_SetString(PyExc_ValueError, "vf_norm shape must match (cx, vy, vz)");
            return NULL;
        }
        norm_total = total;
    } else { // 1D
        norm_total = PyArray_SIZE(vf_norm_obj);
        if (norm_total != total) {
            Py_DECREF(vf_chunk_obj);
            Py_DECREF(vf_norm_obj);
            PyErr_SetString(PyExc_ValueError, "vf_norm length must equal cx*vy*vz when flattened");
            return NULL;
        }
    }

    float *vf_chunk = (float*)PyArray_DATA(vf_chunk_obj);
    float *vf_norm  = (float*)PyArray_DATA(vf_norm_obj);

    // thread count clamp
    int T = ncores;
#ifdef _OPENMP
    T = std::max(1, std::min(ncores, omp_get_max_threads()));
#else
    T = 1;
#endif

    // --- 1) Pass-1: thread-local nonzero counts
    std::vector<long long> tcnt(T, 0);

    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        long long c = 0;

        #pragma omp for schedule(static)
        for (npy_intp idx = 0; idx < total; ++idx) {
            if (vf_norm[idx] > 0.0f) c++;
        }
        tcnt[tid] = c;
    }
#else
    {
        long long c = 0;
        for (npy_intp idx = 0; idx < total; ++idx) if (vf_norm[idx] > 0.0f) c++;
        tcnt[0] = c;
    }
#endif
    Py_END_ALLOW_THREADS

    // prefix sum offsets
    std::vector<long long> off(T, 0);
    long long count = 0;
    for (int t = 0; t < T; ++t) {
        off[t] = count;
        count += tcnt[t];
    }

    // count -> npy_intp range check
    if (count < 0 || (unsigned long long)count > (unsigned long long)std::numeric_limits<npy_intp>::max()) {
        Py_DECREF(vf_chunk_obj);
        Py_DECREF(vf_norm_obj);
        PyErr_SetString(PyExc_OverflowError, "nonzero count exceeds npy_intp capacity");
        return NULL;
    }

    npy_intp out_n = (npy_intp)count;

    // --- allocate outputs (exact size, no giant temporaries)
    npy_intp out_shape1[1] = { out_n };
    PyArrayObject *x_arr = (PyArrayObject*)PyArray_SimpleNew(1, out_shape1, NPY_INT32);
    PyArrayObject *y_arr = (PyArrayObject*)PyArray_SimpleNew(1, out_shape1, NPY_INT32);
    PyArrayObject *z_arr = (PyArrayObject*)PyArray_SimpleNew(1, out_shape1, NPY_INT32);

    npy_intp out_shape2[2] = { out_n, ng };
    PyArrayObject *data_arr = (PyArrayObject*)PyArray_SimpleNew(2, out_shape2, NPY_FLOAT32);

    if (!x_arr || !y_arr || !z_arr || !data_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(z_arr);
        Py_XDECREF(data_arr);
        Py_DECREF(vf_chunk_obj);
        Py_DECREF(vf_norm_obj);
        PyErr_NoMemory();
        return NULL;
    }

    int   *out_x    = (int*)PyArray_DATA(x_arr);
    int   *out_y    = (int*)PyArray_DATA(y_arr);
    int   *out_z    = (int*)PyArray_DATA(z_arr);
    float *out_data = (float*)PyArray_DATA(data_arr);

    // --- 2) Pass-2: fill outputs using per-thread offsets (no atomics, no huge malloc)
    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        long long w = off[tid];

        #pragma omp for schedule(static)
        for (npy_intp idx = 0; idx < total; ++idx) {
            float norm = vf_norm[idx];
            if (norm <= 0.0f) continue;

            // unravel idx -> (ix, iy, iz)
            npy_intp tmp = idx;
            const int iz = (int)(tmp % vz); tmp /= vz;
            const int iy = (int)(tmp % vy);
            const int ix = (int)(tmp / vy);

            const float scale = (float)(size_after_norm / (double)norm);

            out_x[w] = ix + x0;
            out_y[w] = iy;
            out_z[w] = iz;

            float *dst = out_data + (npy_intp)w * ng;
            float *src = vf_chunk + (npy_intp)idx * ng;

            for (npy_intp g = 0; g < ng; ++g) dst[g] = src[g] * scale;

            w++;
        }
        // (optional) sanity: w should end at off[tid] + tcnt[tid]
    }
#else
    {
        long long w = 0;
        for (npy_intp idx = 0; idx < total; ++idx) {
            float norm = vf_norm[idx];
            if (norm <= 0.0f) continue;

            npy_intp tmp = idx;
            const int iz = (int)(tmp % vz); tmp /= vz;
            const int iy = (int)(tmp % vy);
            const int ix = (int)(tmp / vy);

            const float scale = (float)(size_after_norm / (double)norm);

            out_x[w] = ix + x0;
            out_y[w] = iy;
            out_z[w] = iz;

            float *dst = out_data + (npy_intp)w * ng;
            float *src = vf_chunk + (npy_intp)idx * ng;
            for (npy_intp g = 0; g < ng; ++g) dst[g] = src[g] * scale;

            w++;
        }
    }
#endif
    Py_END_ALLOW_THREADS

    // release temporary converted views
    Py_DECREF(vf_chunk_obj);
    Py_DECREF(vf_norm_obj);

    // --- return tuple (steal references => no refcount leak)
    PyObject *ret = PyTuple_New(4);
    if (!ret) {
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(z_arr);
        Py_DECREF(data_arr);
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, (PyObject*)x_arr);    // steals
    PyTuple_SET_ITEM(ret, 1, (PyObject*)y_arr);
    PyTuple_SET_ITEM(ret, 2, (PyObject*)z_arr);
    PyTuple_SET_ITEM(ret, 3, (PyObject*)data_arr);

    return ret;
}


static struct PyMethodDef module_methods[] = {
    {"corr", (PyCFunction)corr, METH_VARARGS, "Calculates Pearson's correlation coefficient."},
    {"calc_ctmap", (PyCFunction)calc_ctmap, METH_VARARGS | METH_KEYWORDS, "Creates a cell type map."},
    {"calc_corrmap", (PyCFunction)calc_corrmap, METH_VARARGS | METH_KEYWORDS, "Creates a correlation map."},
    {"calc_corrmap_2", (PyCFunction)calc_corrmap_2, METH_VARARGS | METH_KEYWORDS, "Creates a correlation map."},
    {"calc_kde", (PyCFunction)calc_kde, METH_VARARGS | METH_KEYWORDS, "Run kernel density estimation."},
    {"flood_fill", (PyCFunction)flood_fill, METH_VARARGS | METH_KEYWORDS, "Performs 3d flood fill based on correlation."},
    {"check_avx512f", (PyCFunction)check_avx512f, METH_NOARGS, "Check if AVX-512F is supported."},
    {"accumulate_gene", (PyCFunction)accumulate_gene, METH_VARARGS | METH_KEYWORDS, "In-place KDE accumulation for one gene using CSR neighbor lists."},
    {"accumulate_gene_vector", (PyCFunction)accumulate_gene_vector, METH_VARARGS | METH_KEYWORDS, "Vectorized in-place KDE accumulation for one gene using CSR neighbor lists."},
    {"accumulate_gene_vector_sparse", (PyCFunction)accumulate_gene_vector, METH_VARARGS | METH_KEYWORDS, "Alias for accumulate_gene_vector to satisfy sparse interface."},
    {"normalize_vf", (PyCFunction)normalize_vf, METH_VARARGS | METH_KEYWORDS, "Normalize gene expression vectors in a chunk and extract non-zero vectors."},
    {"kde_at_points", (PyCFunction)kde_at_points, METH_VARARGS | METH_KEYWORDS, "Compute raw KDE vectors at arbitrary query positions (gather mode). Result matches accumulate_gene_vector at the same coordinates."},
    // {"normalize_vf_sparse", (PyCFunction)normalize_vf_sparse, METH_VARARGS | METH_KEYWORDS, "Normalize gene expression vectors in a chunk and extract non-zero vectors for sparse data."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef_utils = {
        PyModuleDef_HEAD_INIT,
        "analysis_utils",
        NULL,
        -1,
        module_methods
};
#endif

PyMODINIT_FUNC
PyInit_utils(void)
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef_utils);
#else
    Py_InitModule("utils", module_methods);
#endif
    import_array();
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
