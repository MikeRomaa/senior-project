#define MATX_ENABLE_PYBIND11 1
#define SAMPLES_PER_SPECTRA 4603_z
#define SPECTRA_PER_RUN 1_z
#define WIEN_B 2.897771955e-3

#include <matx.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace py = pybind11;

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

constexpr size_t operator"" _z(unsigned long long n) { return n; }

// struct AnalyzeResult {
//     float temperature; // Provides better alignment than using a double. We don't really need double precision here
//     uint32_t elements;
//     uint8_t classification; // Should we be using a bit vector..?
// };

struct get_star_idx {
    __device__
    size_t operator()(size_t idx) const {
        return idx / SAMPLES_PER_SPECTRA;
    }
};

struct max_flux_idx {
    __device__
    thrust::tuple<size_t, double> operator()(const thrust::tuple<size_t, double> &a, const thrust::tuple<size_t, double> &b) {
        if (thrust::get<1>(b) > thrust::get<1>(a)) {
            return b;
        }

        return a;
    }
};

struct get_temperature {
    float first_wavelength;
    float dispersion_per_pixel;

    __host__
    get_temperature(float _first_wavelength, float _dispersion_per_pixel):
        first_wavelength(_first_wavelength),
        dispersion_per_pixel(_dispersion_per_pixel) {}

    __device__
    float operator()(size_t star_idx, size_t idx) const {
        size_t offset = idx - star_idx * SAMPLES_PER_SPECTRA;
        float wavelength = __exp10f(first_wavelength + offset * dispersion_per_pixel);

        return WIEN_B / wavelength;
    }
};

// `py::array::c_style | py::array::forcecast` restricts this to only accept "dense"
// arrays that we can directly reinterpret as a row-major `double*`
//
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
py::array_t<float> temperatures(
    py::array_t<double, py::array::c_style | py::array::forcecast> py_model,
    float first_wavelength,
    float dispersion_per_pixel
) {
    py::buffer_info buf_model = py_model.request();

    assert(buf_model.ndim == 2);
    assert(buf_model.shape[0] == SPECTRA_PER_RUN);
    assert(buf_model.shape[1] == SAMPLES_PER_SPECTRA);

    double* model = reinterpret_cast<double*>(buf_model.ptr);
    thrust::device_vector<double> d_model(model, model + SPECTRA_PER_RUN * SAMPLES_PER_SPECTRA);

    // Calculate temperatures by using Wien's displacement law:
    //
    //      T = b / λ_peak
    //
    // where `b` is Wien's displacement constant, equal to
    //
    //      2.897771955e-3 m*K
    //

    // The star index will act as our key in the following reduction,
    // since we want to get the highest-flux wavelength for EACH star.
    thrust::counting_iterator idx_begin = thrust::make_counting_iterator(0_z);
    thrust::counting_iterator idx_end = thrust::make_counting_iterator(SPECTRA_PER_RUN * SAMPLES_PER_SPECTRA);

    thrust::transform_iterator idx_star_begin = thrust::make_transform_iterator(idx_begin, get_star_idx());
    thrust::transform_iterator idx_star_end = thrust::make_transform_iterator(idx_end, get_star_idx());

    // First we find the index where the maximum flux occurs
    thrust::device_vector<double> d_max_flux_idx(SPECTRA_PER_RUN);
    thrust::reduce_by_key(
        idx_star_begin,
        idx_star_end,
        thrust::make_zip_iterator(thrust::make_tuple(
            idx_begin,
            d_model.begin()
        )),
        thrust::make_discard_iterator(), // We don't care about the index of the star in the output
        thrust::make_zip_iterator(thrust::make_tuple(
            d_max_flux_idx.begin(), 
            thrust::make_discard_iterator() // We don't care about the value of the peak
        )),
        thrust::equal_to<size_t>(),
        max_flux_idx()
    );

    thrust::device_vector<float> d_temperatures(SPECTRA_PER_RUN);
    thrust::transform(
        idx_star_begin,
        idx_star_end,
        d_max_flux_idx.begin(),
        d_temperatures.begin(),
        get_temperature(first_wavelength, dispersion_per_pixel)
    );

    thrust::host_vector<float> temperatures(d_temperatures);

    return py::array_t<float>(
        { SPECTRA_PER_RUN },
        { sizeof(float) },
        thrust::raw_pointer_cast(temperatures.data())
    );
}

// Define the Python FFI bindings
PYBIND11_MODULE(stargaze, m)
{
    m.doc() = "";
    m.def("temperatures", temperatures);
}
