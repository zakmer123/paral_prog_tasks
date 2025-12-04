#ifndef LINEAR_OPERATOR_H
#define LINEAR_OPERATOR_H

class LinearOperator {
public:
    virtual ~LinearOperator() = default;
    virtual void apply(const double* x, double* y) = 0;
};

#endif
