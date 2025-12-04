#ifndef FIELD_H
#define FIELD_H

#include "DomainDecomposition.h"

class Field { 
protected:
    DomainDecomposition& domain_;
    std::vector<double> data_;

    Field(DomainDecomposition& domain) : domain_(domain), data_(domain.localNxNy(), 0) {}
public:
    virtual ~Field() = default;

    virtual double* data() { return data_.data(); }
    virtual const double* data() const { return data_.data(); }
    virtual double& operator[](int index) { return data_[index]; }
    virtual double operator[](int index) const { return data_[index]; }
};

#endif