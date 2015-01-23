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
Opm::Petsc::Petsc handle( &argc, &argvv, NULL, "help" );

using Opm::Petsc::Matrix;
using Opm::Petsc::Vector;

static void have_public_types() {
    Opm::Petsc::Matrix::scalar scalar;
    Opm::Petsc::Matrix::size_type size_type;

    /*
     * Ignore "unused variable" warnings
     */
    (void) scalar;
    (void) size_type;
}

BOOST_AUTO_TEST_CASE(test_PetscVector_coverage) {
    have_public_types();
}

const Matrix::size_type row_ind[] = { 0, 1, 2, 3, 4 };
const Matrix::size_type col_ind[] = { 0, 1, 5, 2, 3 };
const Matrix::scalar values[] = {
    10.0, 0, 0, 0, 0, 5.72,
    0.2, 0, 0, 0, 0, 0,
    0, 0, 4.2, 0, 0, 0,
    0, 0, 0, 3.4, 0, 0,
    0, 0, 3.14, 0, 0, 0,
    0, 0, 0, 0, 0, 0 };

const Matrix::size_type csr_rows[] = { 0, 2, 3, 4, 5, 6, 6 };
const Matrix::size_type csr_cols[] = { 0, 5, 0, 2, 3, 2 };
const Matrix::scalar csr_vals[] = { 10.0, 5.72, 0.2, 4.2, 3.4, 3.14 };
const Matrix::scalar csr_vals_m2[] = { 20.0, 11.44, 0.4, 8.4, 6.8, 6.28 };
const Matrix::scalar csr_vals_zero[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

const Matrix::scalar spmv_vals[] = { 117.961, 2.0, 0.84, 14.28, 0.628, 0.0 };

const Matrix::size_type row_ind_trans[] = { 0, 0, 2, 2, 3, 5 };
const Matrix::size_type col_ind_trans[] = { 0, 1, 2, 4, 3, 0 };
const Matrix::scalar values_trans[] = { 10.0, 0.2, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        0, 0, 4.2, 0, 3.4, 0,
                                        0, 0, 0, 3.14, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        5.72, 0, 0, 0, 0, 0 };

BOOST_AUTO_TEST_CASE(test_compare_builder_raw_petsc) {
    /* ensure ownership transfer and copy construction works as intended by
     * building from raw PETSc functions
     */

    Mat matpetsc;
    MatCreate( PETSC_COMM_WORLD, &matpetsc );
    MatSetSizes( matpetsc, PETSC_DECIDE, PETSC_DECIDE, 10, 10 );
    MatSetFromOptions( matpetsc );
    MatSetUp( matpetsc );
    MatSetValue( matpetsc, 1, 1, 2.3, INSERT_VALUES );
    MatAssemblyBegin( matpetsc, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( matpetsc, MAT_FINAL_ASSEMBLY );
    Matrix mtx( matpetsc );

    Matrix::Builder::Inserter builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );
    auto builder_mtx = Opm::Petsc::commit( builder );

    BOOST_CHECK( identical( builder_mtx, mtx ) );
}

BOOST_AUTO_TEST_CASE(test_Matrix_copies) {
    Matrix::Builder::Inserter builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );

    auto mtx = commit( builder );
    Matrix copy( mtx );

    auto builder_copy = builder;
    auto builder_copy_mtx = commit( builder_copy );

    BOOST_CHECK( identical( mtx, copy ) );
    BOOST_CHECK( identical( builder_copy_mtx, mtx ) );
    BOOST_CHECK( identical( builder_copy_mtx, copy ) );
}

BOOST_AUTO_TEST_CASE(test_matrix_move) {
    Matrix::Builder::Inserter builder( 10, 10 );
    builder.insert( 1, 1, 2.3 );

    auto mtx = Opm::Petsc::commit( builder );
    auto moved = Opm::Petsc::commit( std::move( builder ) );

    BOOST_CHECK( identical( moved, mtx ) );
}

/*
 * A convenient way to build a Matrix (to avoid repetition). Creates a
 * CSR-style Matrix from the constant arrays
 */
static Matrix get_csr_matrix( const Matrix::scalar* vals ) {
    std::vector< Matrix::scalar > csr_val_vec;
    std::vector< Matrix::size_type > csr_row_vec;
    std::vector< Matrix::size_type > csr_col_vec;
    for( int i = 0; i < 6; ++i ) {
        csr_val_vec.push_back( vals[ i ] );
        csr_col_vec.push_back( csr_cols[ i ] );
        csr_row_vec.push_back( csr_rows[ i ] );
    }
    csr_row_vec.push_back( csr_rows[ 6 ] );

    Matrix::Builder::Inserter csr_builder( 6, 6 );
    csr_builder.insert( csr_val_vec, csr_row_vec, csr_col_vec );

    return Opm::Petsc::commit( csr_builder );
}

BOOST_AUTO_TEST_CASE(test_matrixbuilder) {
    auto mtx = get_csr_matrix( csr_vals );

    Matrix::Builder::Inserter direct_builder( 6, 6 );
    direct_builder.insert( 0, 0, 10.0 );
    direct_builder.insert( 0, 5, 5.72 );
    direct_builder.insert( 1, 0, 0.2 );
    direct_builder.insert( 2, 2, 4.2 );
    direct_builder.insert( 3, 3, 3.4 );
    direct_builder.insert( 4, 2, 3.14 );

    auto dmtx = Opm::Petsc::commit( direct_builder );

    BOOST_CHECK( identical( mtx, dmtx ) );
}

BOOST_AUTO_TEST_CASE(test_builder_methods_equivalence) {
    Matrix::Builder::Inserter ins( 6, 6 );
    Matrix::Builder::Accumulator add( 6, 6 );

    ins.insert( 0, 0, 10.0 ).
    insert( 0, 5, 5.72 ).
    insert( 1, 0, 0.2 ).
    insert( 2, 2, 4.2 ).
    insert( 3, 3, 3.4 ).
    insert( 4, 2, 3.14 );

    /* build an identical Matrix, but with add. Notice that both += and -= are
     * used, in the lines with two operations. They sum to the equivalent cell
     * value
     */
    add.add( 0, 0, 10.0 ).
    add( 0, 5, 5.72 ).
    add( 1, 0, 0.1 ).add( 1, 0, 0.1 ).
    add( 2, 2, 6.2 ).add( 2, 2, -2.0 ).
    add( 3, 3, 3.4 ).
    add( 4, 2, 3.14 );

    auto from_ins = Opm::Petsc::commit( ins );
    auto from_add = Opm::Petsc::commit( add );

    BOOST_CHECK( identical( from_ins, from_add ) );
}

BOOST_AUTO_TEST_CASE(test_builder_conversion) {
    Matrix::Builder::Inserter ins( 6, 6 );

    ins.insert( 0, 0, 10.0 ).
    insert( 0, 5, 5.72 ).
    insert( 1, 0, 0.2 ).
    insert( 4, 2, 3.14 );

    Matrix::Builder::Accumulator accer( std::move( ins ) );

    /* build an identical Matrix, but with add. Notice that both += and -= are
     * used, in the lines with two operations. They sum to the equivalent cell
     * value
     */
    accer.
    add( 2, 2, 6.2 ).add( 2, 2, -2.0 ).
    add( 3, 3, 3.4 );

    auto from_add = Opm::Petsc::commit( accer );
    auto reference_matrix = get_csr_matrix( csr_vals );

    BOOST_CHECK( identical( from_add, reference_matrix ) );
}

BOOST_AUTO_TEST_CASE(test_matrix_arithmetic) {
    auto base_matrix = get_csr_matrix( csr_vals );
    auto matrix_m2 = get_csr_matrix( csr_vals_m2 );
    auto base_matrix_copy = base_matrix;
    auto matrix_zero = get_csr_matrix( csr_vals_zero );
    Vector zero_vector( 6, 0.0 );
    Vector base_vector( std::vector< Vector::scalar >( csr_vals, csr_vals + 6 ) );
    Vector spmv_vector( std::vector< Vector::scalar >( spmv_vals, spmv_vals + 6 ) );

    BOOST_CHECK( identical( matrix_m2, base_matrix * 2 ) );
    BOOST_CHECK( identical( matrix_m2, base_matrix + base_matrix ) );
    BOOST_CHECK( identical( matrix_zero, base_matrix - base_matrix ) );

    BOOST_CHECK( identical( base_matrix, base_matrix + matrix_zero ) );
    /* base_matrix * matrix_zero is NOT guaranteed to be structurally identical
     * to matrix_zero, therefore this check cannot be done. This is a Petsc
     * detail may be relaxed in the future.
     * BOOST_CHECK( identical( Matrix_zero, base_matrix * matrix_zero ) );
     */
    BOOST_CHECK( identical( matrix_zero, base_matrix * 0 ) );

    BOOST_CHECK( zero_vector == zero_vector * base_matrix );

    /*
     * Unfortunately, Vector comparison suffers from the same. It is a bitwise
     * equality test, therefore rounding errors are not accounted for and
     * things that would otherwise be equal are not
     * BOOST_CHECK( spmv_vector == base_matrix * base_vector );
     */
}

BOOST_AUTO_TEST_CASE(test_petscvector_transpose) {
    std::vector< Matrix::scalar > val_vec;
    for( int i = 0; i < 6*6; ++i ) {
        val_vec.push_back( values[ i ] );
    }

    Matrix source( val_vec, 6, 6 );
    Matrix inplace_transposed = source;

    auto transposed = transpose( source );
    auto double_transposed = transpose( transposed );
    inplace_transposed.transpose();

    BOOST_CHECK( identical( double_transposed, source ) );
    BOOST_CHECK( identical( transposed, inplace_transposed ) );
}
