#include <config.h>

#include <opm/core/linalg/SparseMatrix.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE MatrixTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

template< typename S >
static void test_public_types() {
    typename Opm::SparseMatrix< S >::scalar scalar;
    typename Opm::SparseMatrix< S >::type type;
    typename Opm::SparseMatrix< S >::size_type size_type;
    typename Opm::SparseMatrix< S >::unmanaged unmanaged;
    typename Opm::SparseMatrix< S >::iterator iterator;
    typename Opm::SparseMatrix< S >::const_iterator const_iterator;
    typename Opm::SparseMatrix< S >::row_ref row_ref;
    typename Opm::SparseMatrix< S >::const_row_ref const_row_ref;

    /*
     * Ignore "unused variable" warnings
     */
    (void) scalar;
    (void) type;
    (void) size_type;
    (void) unmanaged;
    (void) iterator;
    (void) const_iterator;
    (void) row_ref;
    (void) const_row_ref;
}

template< typename S >
static void test_constructors() {
    Opm::SparseMatrix< S > def;
}

template< typename S >
static void test_methods() {
    Opm::SparseMatrixBuilder< double > builder;
    builder.add( 0, 0, 1.0 );

    auto matrix = builder.convert< S >();

    matrix.exists( 0, 0 );
    matrix[ 0 ];
    matrix[ 0 ][ 0 ];

    matrix.begin();
    matrix.end();

    matrix.rows();
    matrix.cols();
    matrix.nonzeros();
}

template< typename S >
static void test_coverage() {
    test_public_types< S >();
    test_constructors< S >();
    test_methods< S >();
}

static void coverage() {
    /* A test to see if the SparseMatrix implementations have implemented all
     * required methods. Add a test_coverage< Storage_type >() and this test
     * will fail to compile if something is missing. Does not test for
     * correctness.
     */

    test_coverage< Opm::CSR< double > >();
}

BOOST_AUTO_TEST_CASE(test_interface) {
    coverage();
}

BOOST_AUTO_TEST_CASE(test_builder_add) {

    Opm::SparseMatrixBuilder< double > builder;

    for( auto i = 0; i < 5; ++i ) {
        for( int j = 0; j < 5; ++j ) 
            builder.add( i, j, 1.0 * i + j );
    }

    auto csr = builder.convert< Opm::CSR< double > >();

    BOOST_CHECK( csr.rows() == 5 );
    BOOST_CHECK( csr.nonzeros() == 5 * 5 );

    BOOST_CHECK( csr[ 0 ][ 0 ] == 0.0 );
    BOOST_CHECK( csr[ 0 ][ 1 ] == 1.0 );
    BOOST_CHECK( csr[ 2 ][ 1 ] == 3.0 );
}

BOOST_AUTO_TEST_CASE(test_builder_sparse) {

    Opm::SparseMatrixBuilder< double > builder;

    builder.add( 1, 4, 1.0 );
    builder.add( 0, 2, 3.0 );
    builder.add( 2, 1, 5.0 );
    builder.add( 4, 3, 2.0 );

    auto csr = builder.convert< Opm::CSR< double > >();

    BOOST_CHECK( csr.rows() == 5 );
    BOOST_CHECK( csr.nonzeros() == 4 );

    BOOST_CHECK( csr[ 1 ][ 4 ] == 1.0 );
    BOOST_CHECK( csr[ 0 ][ 2 ] == 3.0 );
    BOOST_CHECK( csr[ 2 ][ 1 ] == 5.0 );
    BOOST_CHECK( csr[ 4 ][ 3 ] == 2.0 );

    BOOST_CHECK( !csr.exists( 0, 0 ) );
    BOOST_CHECK( csr.exists( 1, 4 ) );
}

