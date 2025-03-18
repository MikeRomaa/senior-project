#define WIEN_B 28980000

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

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

struct max_flux_idx {
    __device__
    thrust::tuple<size_t, double> operator()(const thrust::tuple<size_t, double> &a, const thrust::tuple<size_t, double> &b) {
        return thrust::get<1>(b) > thrust::get<1>(a) ? b : a;
    }
};

struct get_temperature {
    size_t samples_per_spectra;
    float first_wavelength;
    float dispersion_per_pixel;

    __host__
    get_temperature(size_t _samples_per_spectra, float _first_wavelength, float _dispersion_per_pixel):
        samples_per_spectra(_samples_per_spectra),
        first_wavelength(_first_wavelength),
        dispersion_per_pixel(_dispersion_per_pixel) {}

    __device__
    float operator()(size_t star_idx, size_t idx) const {
        size_t offset = idx - star_idx * samples_per_spectra;
        float wavelength = __exp10f(first_wavelength + offset * dispersion_per_pixel);

        return WIEN_B / wavelength;
    }
};

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
py::array_t<float> temperatures(
    py::array_t<double, py::array::c_style | py::array::forcecast> py_model,
    float first_wavelength,
    float dispersion_per_pixel
) {
    py::buffer_info buf_model = py_model.request();

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

    double* model = reinterpret_cast<double*>(buf_model.ptr);
    thrust::device_vector<double> d_model(model, model + buf_size);

    // The star index will act as our key in the following reduction,
    // since we want to get the highest-flux wavelength for EACH star.

    auto idx_begin = thrust::make_counting_iterator<size_t>(0);
    auto idx_end = thrust::make_counting_iterator<size_t>(buf_size);

    auto idx_star_begin = thrust::make_transform_iterator(idx_begin, thrust::placeholders::_1 / samples_per_spectra);
    auto idx_star_end = thrust::make_transform_iterator(idx_end, thrust::placeholders::_1 / samples_per_spectra);

    auto idx_sample_begin = thrust::make_transform_iterator(idx_begin, thrust::placeholders::_1 % samples_per_spectra);

    // First we find the index where the maximum flux occurs
    thrust::device_vector<size_t> d_max_flux_idx(spectra_per_run);
    thrust::reduce_by_key(
        idx_star_begin,
        idx_star_end,
        thrust::make_zip_iterator(thrust::make_tuple(
            idx_sample_begin,
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

    thrust::device_vector<float> d_temperatures(spectra_per_run);
    thrust::transform(
        idx_star_begin,
        idx_star_end,
        d_max_flux_idx.begin(),
        d_temperatures.begin(),
        get_temperature(samples_per_spectra, first_wavelength, dispersion_per_pixel)
    );

    thrust::host_vector<float> temperatures(d_temperatures);

    return py::array_t<float>(
        { spectra_per_run },
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
