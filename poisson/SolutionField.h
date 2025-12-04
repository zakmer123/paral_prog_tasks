#ifndef SOLUTIONFIELD_H
#define SOLUTIONFIELD_H

#include "Field.h"

class SolutionField : public Field {
public:
	SolutionField(DomainDecomposition& domain) : Field(domain) {}

	void initialize();
};

#endif
