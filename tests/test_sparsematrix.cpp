#include <config.h>

#include <opm/core/linalg/SparseMatrix.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE MatrixTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_builder_add) {

    Opm::SparseMatrixBuilder< double > builder;

    for( auto i = 0; i < 5; ++i ) {
        for( int j = 0; j < 5; ++j ) 
            builder.add( i, j, 1.0 * i + j );
    }

    auto csr = builder.to_csr();

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

    auto csr = builder.to_csr();

    BOOST_CHECK( csr.rows() == 5 );
    BOOST_CHECK( csr.nonzeros() == 4 );

    BOOST_CHECK( csr[ 1 ][ 4 ] == 1.0 );
    BOOST_CHECK( csr[ 0 ][ 2 ] == 3.0 );
    BOOST_CHECK( csr[ 2 ][ 1 ] == 5.0 );
    BOOST_CHECK( csr[ 4 ][ 3 ] == 2.0 );

    BOOST_CHECK( !csr.exists( 0, 0 ) );
    BOOST_CHECK( csr.exists( 1, 4 ) );
}
