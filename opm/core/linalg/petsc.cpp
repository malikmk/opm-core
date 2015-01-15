#include <opm/core/linalg/petsc.hpp>

namespace Opm {
namespace petsc {

petsc::petsc(
        int* argc,
        char*** argv,
        const char* file,
        const char* help ) {

    PetscInitialize( argc, argv, file, help );
}

petsc::~petsc() {
    PetscFinalize();
}

}
}
