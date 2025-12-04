#include "SolutionField.h"

void SolutionField::initialize() {
    double hy = rev(domain_.globalNy() + 1);

    for (int j = 0; j < domain_.localNy(); ++j) {
        data_[j] = hy * (j + (domain_.rank() % domain_.pry()) * (domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY()) + (1 - domain_.offsetBottomY()));
    }
    for (int i = 1; i < domain_.localNx(); ++i) {
        //cblas_dcopy(domain_.localNy(), data_.data(), 1, data_.data() + i * domain_.localNy(), 1);
        memcpy(data_.data() + i * domain_.localNy(), data_.data(), sizeof(double) * domain_.localNy());
    }
    return;
}
