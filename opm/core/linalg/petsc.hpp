#ifndef OPM_PETSC_H
#define OPM_PETSC_H

/*
 * This file introduces the Opm::petsc namespace for petsc support, as well as
 * a RAII helper class to manage the lifetime of the petsc database and MPI
 * use.
 */

#define PETSC_CLANGUAGE_CXX 1 //For CHKERRXX macro
#include <petsc.h>

namespace Opm {
namespace petsc {

/*
 * Initialize this class at the start of your program, and pass argc and argv
 * from main, then forget about it. This object must be alive before any other
 * petsc feature is used.
 */

class petsc {
    public:
        petsc( int* argc, char*** argv, const char* file, const char* help );
        ~petsc();
};

}
}

#endif //OPM_PETSC_H