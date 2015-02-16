#include <config.h>

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>
#include <opm/core/linalg/petscmatrix.hpp>
#include <opm/core/linalg/petscsolver.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE PetscSolverTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

using namespace Opm::Petsc;

int argc = 0;
char** argv = 0;
Petsc handle( &argc, &argv, NULL, "help" );

BOOST_AUTO_TEST_CASE( trivial ) {

    Matrix::Builder::Inserter builder( 2, 2 );
    builder.insert( 0, 0, 2 );
    builder.insert( 1, 1, 2 );

    // [ 2, 0 ] [ x1 ] = [ 2 ]
    // [ 0, 2 ] [ x2 ] = [ 1 ]

    Matrix A = commit( std::move( builder ) );

    std::vector< Vector::scalar > vec( 2 );
    vec[ 0 ] = 2; vec[ 1 ] = 1;

    Vector b( vec );
    vec[ 0 ] = 1; vec[ 1 ] = 0.5;
    Vector result( vec );

    auto x = solve( A, b );
    auto y = solve( A, b, A );
    auto z = solve( A, b, A,
            Solver::Pc_type( "sor" ),
            Solver::Ksp_type( "cg" ) );

    BOOST_CHECK( x == y );
    BOOST_CHECK( y == z );
    BOOST_CHECK( z == result );
}
