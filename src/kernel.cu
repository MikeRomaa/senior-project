#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "matx.h"

namespace py = pybind11;

const SAMPLES_PER_SPECTRA = 1; // TODO
const SPECTRA_PER_RUN = 1; //TODO

const double FRAUNHOFER_LINES[] = {
    898.765,  // O2
    822.696,  // O2
    759.370,  // O2
    686.719,  // O2
    656.281,  // H
    627.661,  // O2
    589.592,  // Na
    588.995,  // Na
    587.5618, // He
    546.073,  // Hg
    527.039,  // Fe
    518.362,  // Mg
    517.270,  // Mg
    516.891,  // Fe
    516.733,  // Mg
    495.761,  // Fe
    486.134,  // H
    466.814,  // Fe
    438.355,  // Fe
    434.047,  // H
    430.790,  // Fe
    430.774,  // Ca
    410.175,  // H
    396.847,  // Ca+
    393.368,  // Ca+
    382.044,  // Fe
    358.121,  // Fe
    336.112,  // Ti+
    302.108,  // Fe
    299.444,  // Ni
};

const double WIEN_B = 2.897771955e-3

__global__
void kernel(const double* models, const bool* out_elements) {
    double target_wavelength = FRAUNHOFER_LINES[threadIdx.y];
}

struct AnalyzeResult {
    temperature: double;
    classification: short;
    elements: py::array_t<_>;
}

// `py::array::c_style | py::array::forcecast` restricts this to only accept "dense"
// arrays that we can directly reinterpret as a row-major `double*`
//
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
AnalyzeResult stargaze(
    py::array_t<double, py::array::c_style | py::array::forcecast> py_model,
    double first_wavelength,
    double dispersion_per_pixel
) {
    py::buffer_info buf_model = py_model.request();
    
    assert(
        buf_model.ndim == 2 &&
        buf_model.shape[0] == SPECTRA_PER_RUN &&
        buf_model.shape[1] == SAMPLES_PER_SPECTRA
    );

    double* model = reinterpret_cast<double*>(buf_model.ptr);

    // Calculate temperatures by using Wien's displacement law:
    //
    //      T = b / λ_peak
    //
    // where `b` is Wien's displacement constant, equal to
    //
    //      2.897771955e-3 m*K
    //

    // TODO: Maybe we can use `make_static_tensor`?
    matx::tensor_t<double, SPECTRA_PER_RUN, SAMPLES_PER_SPECTRA> tensor(model);

    matx::tensor_t<matx::index_t, SPECTRA_PER_RUN> max_sample_idx;
    matx::tensor_t<double, SPECTRA_PER_RUN> max_flux;
    matx::tensor_t<double, SPECTRA_PER_RUN> temperature;

    (matx::mtie(max_flux, max_sample_idx) = matx::argmax(tensor)).run();
    cudaDeviceSynchronize();  // MatX operations are all asynchronous, we need to wait for them to be completed

    (temperature = WIEN_B / matx::pow(max_flux, first_wavelength + max_sample_idx * dispersion_per_pixel)).run();
    cudaDeviceSynchronize();

    return {
        .temperature = 0,
        .classification = 0,
        .elements = {},
    };
}

// Define the Python FFI bindings
PYBIND11_MODULE(stargaze, m)
{
    m.doc() = "Docstring";
    m.def("stargaze", stargaze);
}
