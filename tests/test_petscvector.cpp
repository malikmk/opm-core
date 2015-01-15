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
Opm::petsc::petsc handle( &argc, &argv, NULL, "help" );

template< class ForwardIterator, class T >
static inline void iota( ForwardIterator first, ForwardIterator last, T value, T step ) {
    while(first != last) { 
        *first++ = value;
        value += step;
    }
}

template< typename T >
static inline std::vector< T > range( T begin, T end, T step = T( 1 ) ) {
    using namespace std;

    std::vector< T > x( ( end - begin ) / step );
    /* if std::iota is available, the function defined above shouldn't be
     * compiled in and std::iota should be used
     */
    iota( x.begin(), x.end(), begin, step );
    return x;
}

static void have_public_types() {
    typename Opm::petsc::vector::scalar scalar;
    typename Opm::petsc::vector::size_type size_type;

    /*
     * Ignore "unused variable" warnings
     */
    (void) scalar;
    (void) size_type;
}

BOOST_AUTO_TEST_CASE(test_petscvector_coverage) {
    /* Test that the required public types are exposed.
     *
     * Will provide compile error if not
     */
    have_public_types();
}

BOOST_AUTO_TEST_CASE(test_petscvector_constructors) {
    using Opm::petsc::vector;

    /* Create reference vector petsc styles. */
    Vec vec_petsc;
    VecCreate( PETSC_COMM_WORLD, &vec_petsc );
    VecSetSizes( vec_petsc, PETSC_DECIDE, 10 );
    VecSetFromOptions( vec_petsc );

    /* ownership transferring constructor */
    vector vector_petsc( vec_petsc );

    /* test copy constructor */
    auto vector_copy = vector( vector_petsc );

    vector vector_size( 10 );
    vector vector_size_elems( 10, 0.0 );

    /* construction from std::vector */
    std::vector< vector::scalar > std_vec( 10, 0.0 );
    auto stdidx = range( 0, 10 );
    vector vector_from_stdvec( std_vec );
    vector vector_from_stdidx( std_vec, stdidx );

    /* construct from raw arrays */
    double vals[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    int idx[] = { 0, 1, 2, 3, 4 };
    vector vector_from_array( vals, 5 );
    vector vector_from_idx( vals, idx, 5 );

    BOOST_CHECK( vector_copy == vector_petsc );
    BOOST_CHECK( vector_from_stdvec == vector_from_stdidx );
    BOOST_CHECK( vector_from_array == vector_from_idx );

    BOOST_CHECK( !( vector_copy != vector_petsc ) );
    BOOST_CHECK( !( vector_from_stdvec != vector_from_stdidx ) );
    BOOST_CHECK( !( vector_from_array != vector_from_idx ) );
}

BOOST_AUTO_TEST_CASE(test_petscvector_arithmetic) {
    using Opm::petsc::vector;


    vector vec1( range( 0.0, 10.0 ) );
    vector vec_add_target( range( 2.0, 12.0 ) );
    auto vec_add = vec1 + 2;
    auto vec_sub = vec_add_target - 2;

    vector vec_mul_target( range( 0.0, 30.0, 3.0 ) );
    auto vec_mul = vec1 * 3;
    auto vec_div = vec_mul_target / 3;

    BOOST_CHECK( vec_add == vec_add_target );
    BOOST_CHECK( vec_sub == vec1 );
    BOOST_CHECK( vec_mul == vec_mul_target );
    BOOST_CHECK( vec_div == vec1 );
}

BOOST_AUTO_TEST_CASE(test_petscvector_functional) {
    using Opm::petsc::vector;

    vector vec1( range( 0.0, 10.0 ) );
    vector vec2( range( 0.0, 20.0, 2.0 ) );

    BOOST_CHECK( ( vec1 * vec2 ) == 570.0 );
    BOOST_CHECK( Opm::petsc::max( vec1 ) ==  9.0 );
    BOOST_CHECK( Opm::petsc::min( vec1 ) ==  0.0 );
    BOOST_CHECK( Opm::petsc::sum( vec1 ) == 45.0 );
}
