#include "ConductivityField.h"

ConductivityField::ConductivityField(DomainDecomposition& domain, std::string& filename) : Field(domain) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename.data(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Datatype filetype;
    MPI_Type_vector(domain_.localNx(), domain_.localNy(), domain_.globalNy(), MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_Offset offset = (domain_.rank() / domain_.pry()) * (domain_.localNx() - domain_.offsetLeftX() - domain_.offsetRightX()) - domain_.offsetLeftX();
    offset *= domain_.globalNy();
    offset += (domain_.rank() % domain_.pry()) * (domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY()) - domain_.offsetBottomY();
    offset *= sizeof(double);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    MPI_File_read_all(fh, data_.data(), domain_.localNxNy(), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
}

void ConductivityField::considerContrast(double contrastValue) {
    for (int i = 0; i < data_.size(); ++i) {
        data_[i] = (pow(10, -contrastValue) - 1) * data_[i] + 1;
    }
    return;
}

void ConductivityField::ignoreContrast(double contrastValue) {
    for (int i = 0; i < data_.size(); ++i) {
        data_[i] = (data_[i] - 1) / (pow(10, -contrastValue) - 1);
    }
    return;
}

void ConductivityField::resizeConductivityField(int scale) {
    int sglobalNx = scale * domain_.globalNx();
    int sglobalNy = scale * domain_.globalNy();
    int slocalNx = (sglobalNx / domain_.prx()) + domain_.offsetLeftX() + domain_.offsetRightX();
    int slocalNy = (sglobalNy / domain_.pry()) + domain_.offsetBottomY() + domain_.offsetTopY();

    std::vector<double> oldData(data_);
    data_.resize(slocalNx * slocalNy);

    int index, sindex;

    if (domain_.offsetBottomY() == 1) {
        if (domain_.offsetLeftX() == 1) { data_[0] = oldData[0]; }
        for (int sx = 0; sx < scale; ++sx) {
            /*cblas_dcopy(domain_.localNx() - domain_.offsetLeftX() - domain_.offsetRightX(),
                oldData.data() + domain_.offsetLeftX() * domain_.localNy(), domain_.localNy(),
                data_.data() + (sx + domain_.offsetLeftX()) * slocalNy, scale * slocalNy);*/
            int nx = domain_.localNx() - domain_.offsetLeftX() - domain_.offsetRightX();
            int ny = domain_.localNy();
            double* src_col0 = oldData.data() + domain_.offsetLeftX() * ny;
            double* dst_col0 = data_.data() + (sx + domain_.offsetLeftX()) * slocalNy;
            for (int j = 0; j < nx; ++j) {
                std::memcpy(dst_col0 + j * scale * slocalNy,
                    src_col0 + j * ny,
                    ny * sizeof(double));
            }
        }
        index = ind(domain_.localNy(), domain_.localNx() - 1, 0);
        sindex = ind(slocalNy, slocalNx - 1, 0);
        if (domain_.offsetRightX() == 1) { data_[sindex] = oldData[index]; }
    }
    if (domain_.offsetLeftX() == 1) {
        for (int sy = 0; sy < scale; ++sy) {
            /*cblas_dcopy(domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY(),
                oldData.data() + domain_.offsetBottomY(), 1,
                data_.data() + sy + domain_.offsetBottomY(), scale);*/
            int ny = domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY();
            int offset = domain_.offsetBottomY();
            double* src = oldData.data() + offset;
            double* dst = data_.data() + sy + offset;
            for (int j = 0; j < ny; ++j) {
                dst[j * scale] = src[j];
            }
        }
    }
    for (int i = domain_.offsetLeftX(); i < domain_.localNx() - domain_.offsetRightX(); ++i) {
        for (int sx = 0; sx < scale; ++sx) {
            index = ind(domain_.localNy(), i, domain_.offsetBottomY());
            sindex = ind(slocalNy, scale * (i - domain_.offsetLeftX()) + sx + domain_.offsetLeftX(), domain_.offsetBottomY());
            for (int sy = 0; sy < scale; ++sy) {
                /*cblas_dcopy(domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY(),
                    oldData.data() + index, 1,
                    data_.data() + sindex + sy, scale);*/
                int ny = domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY();
                double* src = oldData.data() + index;
                double* dst = data_.data() + sindex + sy;
                for (int j = 0; j < ny; ++j) {
                    dst[j * scale] = src[j];
                }
            }
        }
    }
    if (domain_.offsetRightX() == 1) {
        index = ind(domain_.localNy(), domain_.localNx() - 1, domain_.offsetBottomY());
        sindex = ind(slocalNy, slocalNx - 1, domain_.offsetBottomY());
        for (int sy = 0; sy < scale; ++sy) {
            /*cblas_dcopy(domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY(),
                oldData.data() + index, 1,
                data_.data() + sindex + sy, scale);*/
            int ny = domain_.localNy() - domain_.offsetBottomY() - domain_.offsetTopY();
            double* src = oldData.data() + index;
            double* dst = data_.data() + sindex + sy;
            for (int j = 0; j < ny; ++j) {
                dst[j * scale] = src[j];
            }
        }
    }
    if (domain_.offsetTopY() == 1) {
        index = ind(domain_.localNy(), 0, domain_.localNy() - 1);
        sindex = ind(slocalNy, 0, slocalNy - 1);
        if (domain_.offsetLeftX() == 1) { data_[sindex] = oldData[index]; }
        for (int sx = 0; sx < scale; ++sx) {
            /*cblas_dcopy(domain_.localNx() - domain_.offsetLeftX() - domain_.offsetRightX(),
                oldData.data() + domain_.offsetLeftX() * domain_.localNy() + index, domain_.localNy(),
                data_.data() + (sx + domain_.offsetLeftX()) * slocalNy + sindex, scale * slocalNy);*/
            int nx = domain_.localNx() - domain_.offsetLeftX() - domain_.offsetRightX();
            double* src = oldData.data()
                + domain_.offsetLeftX() * domain_.localNy()
                + index;
            double* dst = data_.data()
                + (sx + domain_.offsetLeftX()) * slocalNy
                + sindex;
            for (int j = 0; j < nx; ++j) {
                dst[j * (scale * slocalNy)] = src[j * domain_.localNy()];
            }
        }
        index = ind(domain_.localNy(), domain_.localNx() - 1, domain_.localNy() - 1);
        sindex = ind(slocalNy, slocalNx - 1, slocalNy - 1);
        if (domain_.offsetRightX() == 1) { data_[sindex] = oldData[index]; }
    }

    return;
}