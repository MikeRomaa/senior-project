#define WIEN_B 28980000

#include <stdexcept>

#include <pybind11/iostream.h>
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

#define LOG(LEVEL, FORMAT, ...) printf("%5s [stargaze::%s:%d] " FORMAT "\n", #LEVEL, __func__, __LINE__ __VA_OPT__(,) __VA_ARGS__)

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

struct Data {
    uint16_t temperature;
};

struct max_flux_idx {
    __device__
    thrust::tuple<size_t, double> operator()(
        const thrust::tuple<size_t, double> &a,
        const thrust::tuple<size_t, double> &b
    ) {
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
    uint16_t operator()(const thrust::tuple<size_t, size_t, double> &values) const {
        size_t star_idx = thrust::get<0>(values);
        size_t idx = thrust::get<1>(values);
        double redshift = thrust::get<2>(values);

        size_t offset = idx - star_idx * samples_per_spectra;
        float wavelength = __exp10f(first_wavelength + offset * dispersion_per_pixel) / (1 + redshift);

        return WIEN_B / wavelength;
    }
};

struct gather_data {
    __device__
    Data operator()(uint16_t temperature) const {
        return {
            .temperature = temperature,
        };
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
py::array_t<Data> temperatures(
    py::array_t<double, py::array::c_style | py::array::forcecast> py_model,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_redshift,
    float first_wavelength,
    float dispersion_per_pixel
) {
    LOG(INFO, "entering function");

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

    LOG(INFO, "spectra_per_run=%zu samples_per_spectra=%zu", spectra_per_run, samples_per_spectra);

    if (buf_redshift.ndim != 1) {
        LOG(ERROR, "buf_redshift.size=%zu", buf_redshift.size);
        throw std::runtime_error("expected `redshift` to be 1-dimensional");
    }

    if (buf_redshift.size != spectra_per_run) {
        LOG(ERROR, "buf_redshift.size=%zu", buf_redshift.size);
        throw std::runtime_error("expected `redshift` to have same dimension on axis 0 as `model`");
    }

    auto start = std::chrono::steady_clock::now();

    double* model = reinterpret_cast<double*>(buf_model.ptr);
    double* redshift = reinterpret_cast<double*>(buf_redshift.ptr);

    thrust::device_vector<double> d_model(model, model + buf_model.size);
    thrust::device_vector<double> d_redshift(redshift, redshift + buf_redshift.size);

    // The star index will act as our key in the following reduction,
    // since we want to get the highest-flux wavelength for EACH star.

    auto idx_begin = thrust::make_counting_iterator<size_t>(0);
    auto idx_end = thrust::make_counting_iterator<size_t>(buf_size);

    auto idx_star_begin = thrust::make_transform_iterator(idx_begin, thrust::placeholders::_1 / samples_per_spectra);
    auto idx_star_end = thrust::make_transform_iterator(idx_end, thrust::placeholders::_1 / samples_per_spectra);

    auto idx_sample_begin = thrust::make_transform_iterator(idx_begin, thrust::placeholders::_1 % samples_per_spectra);

    LOG(INFO, "calling `reduce_by_key`");

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

    LOG(INFO, "calling `transform`");

    thrust::device_vector<uint16_t> d_temperatures(spectra_per_run);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            idx_star_begin,
            d_max_flux_idx.begin(),
            d_redshift.begin()
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            idx_star_end,
            d_max_flux_idx.end(),
            d_redshift.end()
        )),
        d_temperatures.begin(),
        get_temperature(samples_per_spectra, first_wavelength, dispersion_per_pixel)
    );

    LOG(INFO, "calling `transform`");

    thrust::device_vector<Data> d_data(spectra_per_run);
    thrust::transform(
        d_temperatures.begin(),
        d_temperatures.end(),
        d_data.begin(),
        gather_data()
    );

    thrust::host_vector<Data> data(d_data);

    auto end = std::chrono::steady_clock::now();

    LOG(INFO, "finished in %ldms", (end - start) / 1ms);

    return py::array_t<Data>(
        { spectra_per_run },
        { sizeof(Data) },
        thrust::raw_pointer_cast(data.data())
    );
}

// Define the Python FFI bindings
PYBIND11_MODULE(stargaze, m)
{
    PYBIND11_NUMPY_DTYPE(Data, temperature);

    m.doc() = "";
    m.def(
        "temperatures",
        temperatures, 
        py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>()
    );
}
