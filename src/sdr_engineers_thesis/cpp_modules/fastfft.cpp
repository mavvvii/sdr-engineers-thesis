// fastfft_ultra.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fftw3.h>
#include <complex>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace py = pybind11;

class FFTWPlanner {
private:
    std::unordered_map<size_t, fftw_plan> plans;
    std::mutex plan_mutex;

public:
    fftw_plan get_plan(size_t N, bool forward = true) {
        std::lock_guard<std::mutex> lock(plan_mutex);
        size_t key = N | (static_cast<size_t>(forward) << 56);

        auto it = plans.find(key);
        if (it != plans.end()) {
            return it->second;
        }

        fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

        fftw_plan plan = fftw_plan_dft_1d(
            static_cast<int>(N),
            in, out,
            forward ? FFTW_FORWARD : FFTW_BACKWARD,
            FFTW_MEASURE
        );

        fftw_free(in);
        fftw_free(out);

        plans[key] = plan;
        return plan;
    }

    ~FFTWPlanner() {
        for (auto& [key, plan] : plans) {
            fftw_destroy_plan(plan);
        }
    }
};

static FFTWPlanner global_planner;

// ULTRA SZYBIA FFT - bez zbędnych konwersji
py::array_t<std::complex<double>> fast_fft_ultra(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input) {

    py::buffer_info buf = input.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be a 1-D complex array");

    size_t N = buf.shape[0];
    std::complex<double>* in_data = static_cast<std::complex<double>*>(buf.ptr);

    auto result = py::array_t<std::complex<double>>(N);
    auto res_buf = result.request();
    std::complex<double>* out_data = static_cast<std::complex<double>*>(res_buf.ptr);

    // Użyj bezpośrednio danych wejściowych i wyjściowych
    fftw_plan plan = global_planner.get_plan(N, true);
    fftw_execute_dft(plan,
        reinterpret_cast<fftw_complex*>(in_data),
        reinterpret_cast<fftw_complex*>(out_data)
    );

    return result;
}

// FFT z okienkowaniem - zoptymalizowana wersja
py::array_t<std::complex<double>> fast_fft_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input,
    py::array_t<double, py::array::c_style | py::array::forcecast> window) {

    py::buffer_info buf_in = input.request();
    py::buffer_info buf_win = window.request();

    if (buf_in.ndim != 1) throw std::runtime_error("Input must be a 1-D complex array");
    size_t N = buf_in.shape[0];
    if (buf_win.size != N) throw std::runtime_error("Window size must match input size");

    std::complex<double>* in_data = static_cast<std::complex<double>*>(buf_in.ptr);
    double* win_data = static_cast<double*>(buf_win.ptr);

    auto result = py::array_t<std::complex<double>>(N);
    auto res_buf = result.request();
    std::complex<double>* out_data = static_cast<std::complex<double>*>(res_buf.ptr);

    // Stwórz tymczasowy bufor z okienkowaniem
    fftw_complex* temp_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    for (size_t i = 0; i < N; ++i) {
        temp_in[i][0] = in_data[i].real() * win_data[i];
        temp_in[i][1] = in_data[i].imag() * win_data[i];
    }

    fftw_plan plan = global_planner.get_plan(N, true);
    fftw_execute_dft(plan, temp_in, reinterpret_cast<fftw_complex*>(out_data));

    fftw_free(temp_in);
    return result;
}

PYBIND11_MODULE(fastfft, m) {
    m.doc() = "Ultra fast FFT wrapper using FFTW3";

    m.def("fft", &fast_fft_ultra, "Perform ultra fast FFT using FFTW3");
    m.def("fft_windowed", &fast_fft_windowed, "Perform FFT with windowing");
}
