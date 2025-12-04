#include "RightHandSideField.h"

RightHandSideField::RightHandSideField(DomainDecomposition& domain, ConductivityField& conductivity) : Field(domain) {
    double hy = rev(domain_.globalNy() + 1);
    double hy_2 = rev(sqr(hy));

    if (domain_.rank() % domain_.pry() == domain_.pry() - 1) {
        std::vector<double> subConductivity(domain_.localNx());
        //cblas_dcopy(domain_.localNx(), conductivity.data() + domain_.localNy() - 1, domain_.localNy(), subConductivity.data(), 1);
        //transform(subConductivity.begin(), subConductivity.end(), subConductivity.begin(), [](double x) { return rev(x); });
        //cblas_dscal(domain_.localNx(), hy_2, subConductivity.data(), 1);
        //cblas_dcopy(domain_.localNx(), subConductivity.data(), 1, data_.data() + domain_.localNy() - 1, domain_.localNy());
        for (int i = 0; i < domain_.localNx(); ++i) {
            subConductivity[i] = conductivity[(domain_.localNy() - 1) + i * domain_.localNy()];
        }
        for (double& x : subConductivity) {
            x = hy_2 * rev(x);
        }
        for (int i = 0; i < domain_.localNx(); ++i) {
            data_[(domain_.localNy() - 1) + i * domain_.localNy()] = subConductivity[i];
        }
    }
}
