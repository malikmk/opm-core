#include <config.h>

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE MatrixTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include <vector>

int argc = 0;
char** argv = 0;
Opm::Petsc::Petsc handle( &argc, &argv, NULL, "help" );

template< class ForwardIterator, class T >
static inline void iota( ForwardIterator first, ForwardIterator last, T value, T step ) {
    while(first != last) { 
        *first++ = value;
        value += step;
    }
}

template< typename T >
static inline std::vector< T > range( T begin, T end, T step = T( 1 ) ) {
    std::vector< T > x( ( end - begin ) / step );
    /* if std::iota is available, the function defined above shouldn't be
     * compiled in and std::iota should be used
     */
    iota( x.begin(), x.end(), begin, step );
    return x;
}

static void have_public_types() {
    Opm::Petsc::Vector::scalar scalar;
    Opm::Petsc::Vector::size_type size_type;

    /*
     * Ignore "unused variable" warnings
     */
    (void) scalar;
    (void) size_type;
}

BOOST_AUTO_TEST_CASE(test_PetscVector_coverage) {
    /* Test that the required public types are exposed.
     *
     * Will provide compile error if not
     */
    have_public_types();
}

BOOST_AUTO_TEST_CASE(test_PetscVector_constructors) {
    using Opm::Petsc::Vector;

    /* Create reference Vector Petsc styles. */
    Vec vec_Petsc;
    VecCreate( PETSC_COMM_WORLD, &vec_Petsc );
    VecSetSizes( vec_Petsc, PETSC_DECIDE, 10 );
    VecSetFromOptions( vec_Petsc );

    /* ownership transferring constructor */
    Vector Vector_Petsc( vec_Petsc );

    /* test copy constructor */
    auto Vector_copy = Vector( Vector_Petsc );

    Vector Vector_size( 10 );
    Vector Vector_size_elems( 10, 0.0 );

    /* construction from std::vector */
    std::vector< Vector::scalar > std_vec( 10, 0.0 );
    auto stdidx = range( 0, 10 );
    Vector Vector_from_stdvec( std_vec );
    Vector Vector_from_stdidx( std_vec, stdidx );

    BOOST_CHECK( Vector_copy == Vector_Petsc );
    BOOST_CHECK( Vector_from_stdvec == Vector_from_stdidx );

    BOOST_CHECK( !( Vector_copy != Vector_Petsc ) );
    BOOST_CHECK( !( Vector_from_stdvec != Vector_from_stdidx ) );
}

BOOST_AUTO_TEST_CASE(test_PetscVector_arithmetic) {
    using Opm::Petsc::Vector;


    Vector vec1( range( 0.0, 10.0 ) );
    Vector vec_add_target( range( 2.0, 12.0 ) );
    auto vec_add = vec1 + 2;
    auto vec_sub = vec_add_target - 2;

    Vector vec_mul_target( range( 0.0, 30.0, 3.0 ) );
    auto vec_mul = vec1 * 3;
    auto vec_div = vec_mul_target / 3;

    BOOST_CHECK( vec_add == vec_add_target );
    BOOST_CHECK( vec_sub == vec1 );
    BOOST_CHECK( vec_mul == vec_mul_target );
    BOOST_CHECK( vec_div == vec1 );
}

BOOST_AUTO_TEST_CASE(test_PetscVector_functional) {
    using Opm::Petsc::Vector;

    Vector vec1( range( 0.0, 10.0 ) );
    Vector vec2( range( 0.0, 20.0, 2.0 ) );

    BOOST_CHECK( ( vec1 * vec2 ) == 570.0 );
    BOOST_CHECK( Opm::Petsc::max( vec1 ) ==  9.0 );
    BOOST_CHECK( Opm::Petsc::min( vec1 ) ==  0.0 );
    BOOST_CHECK( Opm::Petsc::sum( vec1 ) == 45.0 );
}
