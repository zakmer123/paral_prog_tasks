#ifndef RIGHTHANDSIDEFIELD_H
#define RIGHTHANDSIDEFIELD_H

#include "Field.h"
#include "ConductivityField.h"

class RightHandSideField : public Field {
public:
	RightHandSideField(DomainDecomposition& domain, ConductivityField& conductivity);
};

#endif
