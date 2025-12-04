#ifndef CONDUCTIVITYFIELD_H
#define CONDUCTIVITYFIELD_H

#include "Field.h"

class ConductivityField : public Field {
public:
    ConductivityField(DomainDecomposition& domain, std::string& filename);

    void considerContrast(double contrastValue);
    void ignoreContrast(double contrastValue);

    void resizeConductivityField(int scale);
};

#endif
