#define WIEN_B 28980000

#include <stdexcept>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define LOG(LEVEL, FORMAT, ...) printf("%5s [stargaze::%s:%d] " FORMAT "\n", #LEVEL, __func__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

#define CUDA_CHECK(ERROR_CODE) {                            \
    do {                                                    \
        if (ERROR_CODE != cudaSuccess) {                    \
            LOG(                                            \
                FATAL, "CUDA error %d: %s",                 \
                ERROR_CODE, cudaGetErrorString(ERROR_CODE)  \
            );                                              \
            throw std::runtime_error(                       \
                "CUDA error, see logs for more information" \
            );                                              \
        }                                                   \
    } while (0);                                            \
}

namespace py = pybind11;
using namespace std::literals;

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

__inline__ __device__
double warp_reduce_max(uint mask, double value) {
    for (uint offset = warpSize / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_down_sync(mask, value, offset));
    }

    return value;
}

__inline__ __device__
double block_reduce_max(double value, size_t num_elements) {
    static __shared__ double shared[32];

    uint warp = threadIdx.x / warpSize;
    uint lane = threadIdx.x % warpSize;

    // First we partially reduce each warp within the block
    // With a maximal block size of 32, this gives us at most 32 partial results

    bool active = blockIdx.x * blockDim.x + threadIdx.x < num_elements;
    value = warp_reduce_max(0xFFFFFFFF, active ? value : 0);

    // Each warp writes its partial result to shared memory via the first lane

    if (lane == 0) {
        shared[warp] = value;
    }

    __syncthreads();

    // Our partial results can now fit into a single warp, so we'll overwrite
    // the first warp with the partial results from before

    if (warp == 0) {
        value = shared[lane];

        // No synchronization is needed here, since the warp runs synchronously

        bool active = blockIdx.x * blockDim.x + lane * warpSize < num_elements;
        value = warp_reduce_max(0xFFFFFFFF, active ? value : 0);
    }

    return value;
}

__inline__ __device__
double atomicMax(double* address, double val) {
    double old = *address;
    atomicMax(
        reinterpret_cast<unsigned long long int*>(address),
        *reinterpret_cast<unsigned long long int*>(&val)
    );
    return old;
}

//   Wavelengths ──►                  
//   ┌─────────────────────────────────┐
// S │ ┌───────┐┌───────┐┌───────┐     │
// t │ │ Block ││ Block ││ Block │     │
// a │ ├───────┤├───────┤├───────┤ ... │
// r │ │ x ──► ││ x ──► ││ x ──► │     │
// s │ └───────┘└───────┘└───────┘     │
// │ │ ┌───────┐┌───────┐┌───────┐     │
// │ │ │ Block ││ Block ││ Block │     │
// ▼ │ ├───────┤├───────┤├───────┤ ... │
//   │ │ x ──► ││ x ──► ││ x ──► │     │
//   │ └───────┘└───────┘└───────┘     │
//   │             ...                 │
//   └─────────────────────────────────┘

// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

__global__
void temperature_kernel(
    const size_t samples_per_spectra,
    const float first_wavelength,
    const float dispersion_per_pixel,
    const double* d_model,
    const double* d_redshift,
    double* d_max_flux,
    uint16_t* d_temperature
) {
    uint star_idx = blockIdx.y;
    uint sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sample = d_model[star_idx * samples_per_spectra + sample_idx];

    double block_max = block_reduce_max(sample, samples_per_spectra);

    if (threadIdx.x == 0) {
        atomicMax(d_max_flux + star_idx, block_max);
    }

    __syncthreads();

    if (sample == d_max_flux[star_idx]) {
        double redshift = d_redshift[star_idx];
        double wavelength = __exp10f(first_wavelength + sample_idx * dispersion_per_pixel) / (1 + redshift);

        d_temperature[star_idx] = WIEN_B / wavelength;
    }
}

// Calculate temperatures by using Wien's displacement law:
//
//      T = b / λ_peak
//
// where `b` is Wien's displacement constant, equal to
//
//      28,980,000 Å*K
//
// The parameter type `py::array::c_style | py::array::forcecast` restricts this to only
// accept "dense" arrays that we can directly reinterpret as a row-major `double*`
//
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
py::array_t<uint16_t> temperatures(
    py::array_t<double, py::array::c_style | py::array::forcecast> py_model,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_redshift,
    float first_wavelength,
    float dispersion_per_pixel
) {
    LOG(INFO, "entering function");
    
    auto start = std::chrono::high_resolution_clock::now();

    py::buffer_info buf_model = py_model.request();
    py::buffer_info buf_redshift = py_redshift.request();

    size_t spectra_per_run;
    size_t samples_per_spectra;
    size_t buf_size = buf_model.size;

    if (buf_model.ndim == 2) {
        spectra_per_run = buf_model.shape[0];
        samples_per_spectra = buf_model.shape[1];
    } else {
        spectra_per_run = 1;
        samples_per_spectra = buf_model.shape[0];
    }

    LOG(INFO, "spectra_per_run=%zu, samples_per_spectra=%zu", spectra_per_run, samples_per_spectra);

    if (buf_redshift.ndim != 1) {
        LOG(ERROR, "buf_redshift.size=%zu", buf_redshift.size);
        throw std::runtime_error("expected `redshift` to be 1-dimensional");
    }

    if (buf_redshift.size != spectra_per_run) {
        LOG(ERROR, "buf_redshift.size=%zu", buf_redshift.size);
        throw std::runtime_error("expected `redshift` to have same dimension on axis 0 as `model`");
    }

    double* model = reinterpret_cast<double*>(buf_model.ptr);
    double* redshift = reinterpret_cast<double*>(buf_redshift.ptr);

    double* d_model;
    double* d_redshift;

    CUDA_CHECK( cudaMalloc(&d_model, buf_model.size * sizeof(double)) );
    CUDA_CHECK( cudaMalloc(&d_redshift, buf_redshift.size * sizeof(double)) );
    CUDA_CHECK( cudaMemcpy(d_model, model, buf_model.size * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_redshift, redshift, buf_redshift.size * sizeof(double), cudaMemcpyHostToDevice) );

    double* d_max_flux;
    uint16_t* d_temperature;

    CUDA_CHECK( cudaMalloc(&d_max_flux, spectra_per_run * sizeof(double)) );
    CUDA_CHECK( cudaMalloc(&d_temperature, spectra_per_run * sizeof(uint16_t)) );

    size_t blocks_per_spectra = ceil((float) samples_per_spectra / 1024);

    auto kernel_start = std::chrono::high_resolution_clock::now();

    temperature_kernel<<<dim3(blocks_per_spectra, spectra_per_run), 1024>>>(
        samples_per_spectra,
        first_wavelength,
        dispersion_per_pixel,
        d_model,
        d_redshift,
        d_max_flux,
        d_temperature
    );

    auto kernel_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK( cudaDeviceSynchronize() );
    CUDA_CHECK( cudaGetLastError() );

    LOG(INFO, "kernel finished in %ldµs", (kernel_end - kernel_start) / 1us);

    uint16_t temperature[spectra_per_run];

    CUDA_CHECK( cudaMemcpy(&temperature, d_temperature, spectra_per_run * sizeof(uint16_t), cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_model) );
    CUDA_CHECK( cudaFree(d_redshift) );
    CUDA_CHECK( cudaFree(d_max_flux) );
    CUDA_CHECK( cudaFree(d_temperature) );

    auto end = std::chrono::high_resolution_clock::now();

    LOG(INFO, "finished in %ldµs", (kernel_end - kernel_start) / 1us);

    return py::array_t<uint16_t>(
        { spectra_per_run },
        { sizeof(uint16_t) },
        temperature
    );
}

// Define the Python FFI bindings
PYBIND11_MODULE(stargaze, m)
{
    m.doc() = "";
    m.def(
        "temperatures",
        temperatures, 
        py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>()
    );
}
