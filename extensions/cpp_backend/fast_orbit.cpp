/*
 * Fast Orbit Implementation
 */

#include "fast_orbit.h"
#include <algorithm>

FractalDynamics2D::FractalDynamics2D(
    const Matrix2& A,
    const Matrix2& B,
    const Matrix2& W,
    const Vector2& c
) : A_(A), B_(B), W_(W), c_(c) {}

FractalDynamics2D::Vector2 FractalDynamics2D::matmul(
    const Matrix2& M,
    const Vector2& v
) const {
    return {
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1]
    };
}

FractalDynamics2D::Vector2 FractalDynamics2D::tanh_vec(const Vector2& v) const {
    return {std::tanh(v[0]), std::tanh(v[1])};
}

FractalDynamics2D::Vector2 FractalDynamics2D::step(const Vector2& x) const {
    // Linear term: A x
    Vector2 linear = matmul(A_, x);

    // Nonlinear term: B tanh(W x)
    Vector2 wx = matmul(W_, x);
    Vector2 tanh_wx = tanh_vec(wx);
    Vector2 nonlinear = matmul(B_, tanh_wx);

    // x_next = linear + nonlinear + c
    return {
        linear[0] + nonlinear[0] + c_[0],
        linear[1] + nonlinear[1] + c_[1]
    };
}

std::vector<FractalDynamics2D::Vector2> FractalDynamics2D::simulate(
    const Vector2& x0,
    int n_steps
) const {
    std::vector<Vector2> trajectory;
    trajectory.reserve(n_steps);

    Vector2 x = x0;
    trajectory.push_back(x);

    for (int i = 1; i < n_steps; ++i) {
        x = step(x);
        trajectory.push_back(x);
    }

    return trajectory;
}
