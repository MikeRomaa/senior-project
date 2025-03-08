#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#define DEG_TO_RAD 0.01745329251
#define EARTH_RADIUS_KM 6371.0088
#define THREADS_PER_BLOCK 1024

namespace py = pybind11;
    
const FRAUNHOFER_LINES = [
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
]

__global__
void haversine(const double* model) {
    int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    
    double phi1 = y1[idx] * DEG_TO_RAD;
    double phi2 = y2[idx] * DEG_TO_RAD;
    double lambda1 = x1[idx] * DEG_TO_RAD;
    double lambda2 = x2[idx] * DEG_TO_RAD;
    
    double d_phi = phi2 - phi1;
    double d_lambda = lambda2 - lambda1;
    
    double hav = (1 - cos(d_phi) + cos(phi1) * cos(phi2) * (1 - cos(d_lambda))) / 2;
    
    out[idx] = 2 * EARTH_RADIUS_KM * asin(sqrt(hav));
}

struct AnalyzeResult {
    temperature: double;
    classification: short;
    elements: py::array_t<_>;
}

AnalyzeResult analyze(py::array_t<double> py_model) {
    py::buffer_info buf_model = py_model.request();
    
    // Copy the input data to the device
    double* model = reinterpret_cast<double*>(buf_model.ptr);
    thrust::device_vector<double> d_model(model, model + buf_model.size);
    
    // Run the kernel function
    int blockSize = ceil((float) buf_model.size / THREADS_PER_BLOCK);
    haversine<<<blockSize, THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_model.data()));
    
    // Allocate new Python array and copy results
    py::array_t<double> py_out;
    py::buffer_info buf_out = py_out.request();
    
    double* out = reinterpret_cast<double*>(buf_out.ptr);
    thrust::copy(d_out.begin(), d_out.end(), out);
    
    return {
        .temperature = 0,
        .classification = 0,
        .elements = out,
    };
}

// Define the Python FFI bindings
PYBIND11_MODULE(haversine, m)
{
    m.doc() = "Docstring";
    m.def("analyze", analyze);
}