#include <config.h>

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmatrix.hpp>
#include <opm/core/linalg/petscvector.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE MatrixTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include <vector>

int argc = 0;
const char* argv[] = { "-on_error_attach_debugger", "gdb", "-start_in_debugger" };
char** argvv = (char**)argv;
Opm::petsc::petsc handle( &argc, &argvv, NULL, "help" );

using Opm::petsc::matrix;
using Opm::petsc::vector;

static void have_public_types() {
    typename Opm::petsc::matrix::scalar scalar;
    typename Opm::petsc::matrix::size_type size_type;

    /*
     * Ignore "unused variable" warnings
     */
    (void) scalar;
    (void) size_type;
}

BOOST_AUTO_TEST_CASE(test_petscvector_coverage) {
    have_public_types();
}

const matrix::size_type row_ind[] = { 0, 1, 2, 3, 4 };
const matrix::size_type col_ind[] = { 0, 1, 5, 2, 3 };
const matrix::scalar values[] = {
    10.0, 0, 0, 0, 0, 5.72,
    0.2, 0, 0, 0, 0, 0,
    0, 0, 4.2, 0, 0, 0,
    0, 0, 0, 3.4, 0, 0,
    0, 0, 3.14, 0, 0, 0,
    0, 0, 0, 0, 0, 0 };

const matrix::size_type csr_rows[] = { 0, 2, 3, 4, 5, 6, 6 };
const matrix::size_type csr_cols[] = { 0, 5, 0, 2, 3, 2 };
const matrix::scalar csr_vals[] = { 10.0, 5.72, 0.2, 4.2, 3.4, 3.14 };
const matrix::scalar csr_vals_m2[] = { 20.0, 11.44, 0.4, 8.4, 6.8, 6.28 };
const matrix::scalar csr_vals_zero[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

const matrix::scalar spmv_vals[] = { 117.961, 2.0, 0.84, 14.28, 0.628, 0.0 };

const matrix::size_type row_ind_trans[] = { 0, 0, 2, 2, 3, 5 };
const matrix::size_type col_ind_trans[] = { 0, 1, 2, 4, 3, 0 };
const matrix::scalar values_trans[] = { 10.0, 0.2, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        0, 0, 4.2, 0, 3.4, 0,
                                        0, 0, 0, 3.14, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        5.72, 0, 0, 0, 0, 0 };

BOOST_AUTO_TEST_CASE(test_compare_builder_raw_petsc) {
    /* ensure ownership transfer and copy construction works as intended by
     * building from raw PETSc functions
     */

    Mat mat_petsc;
    MatCreate( PETSC_COMM_WORLD, &mat_petsc );
    MatSetSizes( mat_petsc, PETSC_DECIDE, PETSC_DECIDE, 10, 10 );
    MatSetFromOptions( mat_petsc );
    MatSetUp( mat_petsc );
    MatSetValue( mat_petsc, 1, 1, 2.3, INSERT_VALUES );
    MatAssemblyBegin( mat_petsc, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( mat_petsc, MAT_FINAL_ASSEMBLY );
    matrix mtx( mat_petsc );

    matrix::builder builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );
    matrix builder_mtx( builder );

    BOOST_CHECK( identical( builder_mtx, mtx ) );
}

BOOST_AUTO_TEST_CASE(test_matrix_copies) {
    matrix::builder builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );

    matrix mtx( builder );
    matrix copy( mtx );

    matrix::builder builder_copy( builder );
    matrix builder_copy_mtx( builder_copy );

    BOOST_CHECK( identical( mtx, copy ) );
    BOOST_CHECK( identical( builder_copy_mtx, mtx ) );
    BOOST_CHECK( identical( builder_copy_mtx, copy ) );
}

BOOST_AUTO_TEST_CASE(test_matrix_move) {
    matrix::builder builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );

    matrix mtx( builder );
    matrix moved( std::move( builder ) );

    BOOST_CHECK( identical( moved, mtx ) );
}

/*
 * A convenient way to build a matrix (to avoid repetition). Creates a
 * CSR-style matrix from the constant arrays
 */
static matrix get_csr_matrix( const matrix::scalar* vals ) {
    std::vector< matrix::scalar > csr_val_vec;
    std::vector< matrix::size_type > csr_row_vec;
    std::vector< matrix::size_type > csr_col_vec;
    for( int i = 0; i < 6; ++i ) {
        csr_val_vec.push_back( vals[ i ] );
        csr_col_vec.push_back( csr_cols[ i ] );
        csr_row_vec.push_back( csr_rows[ i ] );
    }
    csr_row_vec.push_back( csr_rows[ 6 ] );

    matrix::builder csr_builder( 6, 6 );
    csr_builder.insert( csr_val_vec, csr_row_vec, csr_col_vec );

    return csr_builder;
}

BOOST_AUTO_TEST_CASE(test_matrixbuilder) {
    auto mtx = get_csr_matrix( csr_vals );

    matrix::builder direct_builder( 6, 6 );
    direct_builder.insert( 0, 0, 10.0 );
    direct_builder.insert( 0, 5, 5.72 );
    direct_builder.insert( 1, 0, 0.2 );
    direct_builder.insert( 2, 2, 4.2 );
    direct_builder.insert( 3, 3, 3.4 );
    direct_builder.insert( 4, 2, 3.14 );

    auto dmtx = direct_builder.commit();

    BOOST_CHECK( identical( mtx, dmtx ) );
}

BOOST_AUTO_TEST_CASE(test_matrix_arithmetic) {
    auto base_matrix = get_csr_matrix( csr_vals );
    auto matrix_m2 = get_csr_matrix( csr_vals_m2 );
    auto base_matrix_copy = base_matrix;
    auto matrix_zero = get_csr_matrix( csr_vals_zero );
    vector zero_vector( 6, 0.0 );
    vector base_vector( csr_vals, 6 );
    vector spmv_vector( spmv_vals, 6 );

    BOOST_CHECK( identical( matrix_m2, base_matrix * 2 ) );
    BOOST_CHECK( identical( matrix_m2, base_matrix + base_matrix ) );
    BOOST_CHECK( identical( matrix_zero, base_matrix - base_matrix ) );

    BOOST_CHECK( identical( base_matrix, base_matrix + matrix_zero ) );
    /* base_matrix * matrix_zero is NOT guaranteed to be structurally identical
     * to matrix_zero, therefore this check cannot be done. This is a petsc
     * detail may be relaxed in the future.
     * BOOST_CHECK( identical( matrix_zero, base_matrix * matrix_zero ) );
     */
    BOOST_CHECK( identical( matrix_zero, base_matrix * 0 ) );

    BOOST_CHECK( zero_vector == zero_vector * base_matrix );

    /*
     * Unfortunately, vector comparison suffers from the same. It is a bitwise
     * equality test, therefore rounding errors are not accounted for and
     * things that would otherwise be equal are not
     * BOOST_CHECK( spmv_vector == base_matrix * base_vector );
     */
}

BOOST_AUTO_TEST_CASE(test_petscvector_transpose) {
    std::vector< matrix::scalar > val_vec;
    for( int i = 0; i < 6*6; ++i ) {
        val_vec.push_back( values[ i ] );
    }
    matrix source( val_vec, 6, 6 );
    matrix inplace_transposed = source;

    auto transposed = transpose( source );
    auto double_transposed = transpose( transposed );
    inplace_transposed.transpose();

    BOOST_CHECK( identical( double_transposed, source ) );
    BOOST_CHECK( identical( transposed, inplace_transposed ) );
}
