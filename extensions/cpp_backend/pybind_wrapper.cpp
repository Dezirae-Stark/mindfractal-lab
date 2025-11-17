/*
 * Pybind11 Python Bindings
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "fast_orbit.h"

namespace py = pybind11;

py::array_t<double> simulate_orbit_wrapper(
    py::array_t<double> A_arr,
    py::array_t<double> B_arr,
    py::array_t<double> W_arr,
    py::array_t<double> c_arr,
    py::array_t<double> x0_arr,
    int n_steps
) {
    // Extract data from NumPy arrays
    auto A_buf = A_arr.request();
    auto B_buf = B_arr.request();
    auto W_buf = W_arr.request();
    auto c_buf = c_arr.request();
    auto x0_buf = x0_arr.request();

    double* A_ptr = static_cast<double*>(A_buf.ptr);
    double* B_ptr = static_cast<double*>(B_buf.ptr);
    double* W_ptr = static_cast<double*>(W_buf.ptr);
    double* c_ptr = static_cast<double*>(c_buf.ptr);
    double* x0_ptr = static_cast<double*>(x0_buf.ptr);

    // Convert to C++ structures
    FractalDynamics2D::Matrix2 A = {{
        {{A_ptr[0], A_ptr[1]}},
        {{A_ptr[2], A_ptr[3]}}
    }};
    FractalDynamics2D::Matrix2 B = {{
        {{B_ptr[0], B_ptr[1]}},
        {{B_ptr[2], B_ptr[3]}}
    }};
    FractalDynamics2D::Matrix2 W = {{
        {{W_ptr[0], W_ptr[1]}},
        {{W_ptr[2], W_ptr[3]}}
    }};
    FractalDynamics2D::Vector2 c = {c_ptr[0], c_ptr[1]};
    FractalDynamics2D::Vector2 x0 = {x0_ptr[0], x0_ptr[1]};

    // Create model and simulate
    FractalDynamics2D model(A, B, W, c);
    auto trajectory = model.simulate(x0, n_steps);

    // Convert back to NumPy array
    auto result = py::array_t<double>({n_steps, 2});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    for (int i = 0; i < n_steps; ++i) {
        result_ptr[i * 2] = trajectory[i][0];
        result_ptr[i * 2 + 1] = trajectory[i][1];
    }

    return result;
}

PYBIND11_MODULE(fast_orbit, m) {
    m.doc() = "Fast C++ backend for MindFractal Lab orbit simulation";

    m.def("simulate_orbit", &simulate_orbit_wrapper,
          "Simulate orbit using optimized C++ backend",
          py::arg("A"), py::arg("B"), py::arg("W"),
          py::arg("c"), py::arg("x0"), py::arg("n_steps"));
}
