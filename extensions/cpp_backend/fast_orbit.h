/*
 * Fast Orbit Computation (C++)
 *
 * Optimized implementation of fractal dynamics simulation.
 */

#ifndef FAST_ORBIT_H
#define FAST_ORBIT_H

#include <vector>
#include <array>
#include <cmath>

class FractalDynamics2D {
public:
    using Vector2 = std::array<double, 2>;
    using Matrix2 = std::array<std::array<double, 2>, 2>;

    FractalDynamics2D(
        const Matrix2& A,
        const Matrix2& B,
        const Matrix2& W,
        const Vector2& c
    );

    Vector2 step(const Vector2& x) const;

    std::vector<Vector2> simulate(const Vector2& x0, int n_steps) const;

private:
    Matrix2 A_, B_, W_;
    Vector2 c_;

    Vector2 matmul(const Matrix2& M, const Vector2& v) const;
    Vector2 tanh_vec(const Vector2& v) const;
};

#endif // FAST_ORBIT_H
