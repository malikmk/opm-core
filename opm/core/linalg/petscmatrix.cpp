#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>
#include <opm/core/linalg/petscmatrix.hpp>


#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include <iostream>

/* Slight hack to get things working. This sets the communicator used when
 * constructing new vectors - however, this should be configurable. When full
 * petsc support is figured out this will be changed.
 */
static inline MPI_Comm get_comm() { return PETSC_COMM_WORLD; }

template< typename T = Opm::petsc::matrix::size_type, typename U >
static inline std::vector< T > range( U begin, U end ) {

    std::vector< T > x( end - begin );
    /* if std::iota is available, the function defined above shouldn't be
     * compiled in and std::iota should be used
     */
    iota( x.begin(), x.end(), begin );
    return x;
}

namespace Opm {
namespace petsc {

/*
 * Trivial translation between the strongly typed enum and the
 * implicit-convertible MatStructure enum.
 */
static inline MatStructure
nz_structure( matrix::nonzero_pattern x ) {
    switch( x ) {
        case matrix::nonzero_pattern::different:
            return DIFFERENT_NONZERO_PATTERN;

        case matrix::nonzero_pattern::subset:
            return SUBSET_NONZERO_PATTERN;

        case matrix::nonzero_pattern::same:
            return SAME_NONZERO_PATTERN;
    }

    assert( false );
    /* To silence non-void without return statement warning. This should never
     * be reached anyways, and is a sure error.
     */
    throw;
}

static inline Mat default_matrix() {
    /* Meant to be used by other constructors. */
    Mat x;
    MatCreate( get_comm(), &x );
    MatSetFromOptions( x );

    return x;
}

static inline Mat copy_matrix( Mat x ) {
    Mat y;
    auto err = MatDuplicate( x, MAT_COPY_VALUES, &y ); CHKERRXX( err );
    std::unique_ptr< _p_Mat, deleter< _p_Mat > > result( y );
    err = MatCopy( x, y, SAME_NONZERO_PATTERN ); CHKERRXX( err );

    return result.release();
}

static inline
Mat sized_matrix( matrix::size_type rows, matrix::size_type cols ) {
    std::unique_ptr< _p_Mat, deleter< _p_Mat > > result( default_matrix() );
    auto err = MatSetSizes( result.get(),
            PETSC_DECIDE, PETSC_DECIDE,
            rows, cols );
    CHKERRXX( err );

    return result.release();
}

matrix::matrix( const matrix& x ) : uptr( copy_matrix( x ) ) {}

matrix::matrix( const matrix::builder& builder ) :
    uptr( std::move( builder.commit() ) ) {}

matrix::matrix( matrix::builder&& builder ) :
    uptr( std::move( builder.assemble() ) ) {}

matrix::matrix( const std::vector< matrix::scalar >& values,
                matrix::size_type rows,
                matrix::size_type cols ) :
        uptr( sized_matrix( rows, cols ) )
{
    /* this is a very specific override to create a dense matrix */
    MatSetType( this->ptr(), MATSEQDENSE );

    auto err = MatSeqDenseSetPreallocation( this->ptr(), NULL );
    CHKERRXX( err );

    const auto indices = range( 0, std::max( rows, cols ) );

    err = MatSetValues( this->ptr(),
                    rows, indices.data(),
                    cols, indices.data(),
                    values.data(), INSERT_VALUES );
    CHKERRXX( err );

    err = MatAssemblyBegin( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
}

matrix::size_type matrix::rows() const {
    PetscInt x;
    auto err = MatGetSize( this->ptr(), &x, NULL ); CHKERRXX( err );
    return x;
}

matrix::size_type matrix::cols() const {
    PetscInt x;
    auto err = MatGetSize( this->ptr(), NULL, &x ); CHKERRXX( err );
    return x;
}

matrix& matrix::operator*=( matrix::scalar rhs ) {
    auto err = MatScale( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

matrix& matrix::operator/=( matrix::scalar rhs ) {
    return *this *= ( 1 / rhs );
}

matrix& matrix::operator+=( const matrix& rhs ) {
    return this->axpy( rhs, 1, nonzero_pattern::different );
}

matrix& matrix::operator-=( const matrix& rhs ) {
    return this->axpy( rhs, -1, nonzero_pattern::different );
}

matrix& matrix::operator*=( const matrix& rhs ) {
    return this->multiply( rhs );
}

matrix operator*( matrix lhs, matrix::scalar rhs ) {
    return lhs *= rhs;
}

matrix operator*( matrix::scalar lhs, matrix rhs ) {
    return rhs *= lhs;
}

matrix operator/( matrix lhs, matrix::scalar rhs ) {
    return lhs /= rhs;
}

matrix operator+( matrix lhs, const matrix& rhs ) {
    return lhs += rhs;
}

matrix operator-( matrix lhs, const matrix& rhs ) {
    return lhs -= rhs;
}

matrix operator*( const matrix& lhs, const matrix& rhs ) {
    return multiply( lhs, rhs );
}

vector operator*( const matrix& lhs, const vector& rhs ) {
    return multiply( lhs, rhs );
}

vector operator*( const vector& lhs, const matrix& rhs ) {
    return multiply( rhs, lhs );
}

bool identical( const matrix& lhs, const matrix& rhs ) {
    /* Comparing a matrix to itself means they're obviously identical */
    if( lhs.ptr() == rhs.ptr() ) return true;

    /* PETSc throws an exception if matrices are of different sizes. Because
     * two matrices of different sizes -cannot- be equal, we check this first
     */
    PetscInt row_lhs, col_lhs, row_rhs, col_rhs;
    auto err = MatGetSize( lhs, &row_lhs, &col_lhs ); CHKERRXX( err );
    err = MatGetSize( rhs, &row_rhs, &col_rhs ); CHKERRXX( err );

    if( row_lhs != row_rhs ) return false;
    if( col_lhs != col_rhs ) return false;

    /* MatEqual also considers structure when testing for equality. See:
     * http://lists.mcs.anl.gov/pipermail/petsc-users/2015-January/024059.html
     */
    PetscBool eq;
    err = MatEqual( lhs, rhs, &eq ); CHKERRXX( err );
    return eq;
}

matrix& matrix::axpy(
        const matrix& x,
        matrix::scalar a,
        matrix::nonzero_pattern nz ) {

    auto err = MatAXPY( this->ptr(), a, x, nz_structure( nz ) );
    CHKERRXX( err );

    return *this;
}

matrix& matrix::xpy( const matrix& x, matrix::nonzero_pattern nz ) {
    return this->axpy( x, 1, nz );
}

matrix& matrix::multiply( const matrix& x ) {
    auto err = MatMatMult( this->ptr(), x,
            MAT_REUSE_MATRIX, PETSC_DEFAULT, NULL );
    CHKERRXX( err );

    return *this;
}

matrix& matrix::transpose() {
    Mat handle = this->ptr();
    auto err = MatTranspose( handle, MAT_REUSE_MATRIX, &handle );
    CHKERRXX( err );

    return *this;
}

matrix& matrix::hermitian_transpose() {
    auto handle = this->ptr();
    auto err = MatHermitianTranspose( handle,
            MAT_REUSE_MATRIX, &handle );
    CHKERRXX( err );

    return *this;
}

matrix multiply( const matrix& rhs, const matrix& lhs, matrix::scalar fill ) {
    Mat x;
    MatMatMult( rhs, lhs, MAT_INITIAL_MATRIX, fill, &x );

    return matrix( x );
}

vector multiply( const matrix& lhs, const vector& rhs ) {
    Vec x;
    auto err = VecDuplicate( rhs, &x ); CHKERRXX( err );
    MatMult( lhs, rhs, x );
    return vector( x );
}

matrix transpose( const matrix& rhs ) {
    Mat x;
    MatTranspose( rhs, MAT_INITIAL_MATRIX, &x );
    return matrix( x );
}

matrix hermitian_transpose( matrix rhs ) {
    return rhs.hermitian_transpose();
}

matrix::builder::builder( matrix::size_type rows, matrix::size_type cols )
    : matrix( sized_matrix( rows, cols ) )
{

    /*
     * This is an awkward case - we haven't been given much preallocation
     * information to use, so we guess. Using the builder without setting
     * proper preallocation may be considered an error in the future.
     */

    /* first, to avoid overflow, rescale the matrix dimensions to 1% of its
     * size. If the matrix is smaller than 10x10 (unlikely, except for test
     * cases), just set the new dimensions to 1.
     */
    const int m = std::max( 1, rows / 10 );
    const int n = std::max( 1, cols / 10 );

    /*
     * Guess nonzeros per row in the diagonal portion of the matrix. Consider
     * the matrix
     * A | B | C
     * --+---+---
     * D | E | F
     * --+---+---
     * G | H | I
     *
     * Where the letters are sub matrices. The diagonal portion are the
     * matrices A, E and I.
     */

    /*
     * Per row in the diagonal portion we're again assuming 1% fill -of the 1%
     * matrix-, i.e. for a 100k row matrix, we're assuming 10 nonzeros per row.
     * Falling back to a minimum of 1. We're assuming we have the same number
     * off the diagonal as well.
     */
    const int nnz_on_diag = std::max( 1, ( m * n ) / 100 );

    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            2 * nnz_on_diag, NULL );
    CHKERRXX( err );

    err = MatMPIAIJSetPreallocation( this->ptr(),
            nnz_on_diag, NULL,
            nnz_on_diag, NULL );

    CHKERRXX( err );
}

matrix::builder::builder(
        matrix::size_type rows,
        matrix::size_type cols,
        const std::vector< matrix::size_type >& nnz_per_row )
    : matrix( sized_matrix( rows, cols ) )
{

    assert( nnz_per_row.size() == rows );

    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            /*ignored*/ 0,
            nnz_per_row.data() );
    CHKERRXX( err );

    /* Some shenanigans to minimise communication during matrix assembly.
     *
     * It is now sufficient to extract information for the parts that will end
     * up belonging to this processor. In the worst case - the sequential one -
     * it ends up being one full array copy.
     */

    matrix::size_type begin, end;
    MatGetOwnershipRange( this->ptr(), &begin, &end );

    std::vector< matrix::size_type > nnz_diag(
            nnz_per_row.begin() + begin,
            nnz_per_row.begin() + end );

    /*
     * This scheme overcommits local memory by a factor of 2 - that is, if the
     * nnz_per_row vector is precise, preallocation will allocate exactly twice
     * the memory, because it assumes as many nonzeros per row in the
     * off-diagonals as in the diagonals. If this assumption does not hold and
     * you want less memory strain, use a different constructor and give it
     * more information.
     */
    err = MatMPIAIJSetPreallocation( this->ptr(),
            /* ignored */ 0, nnz_diag.data(),
            /* ignored */ 0, nnz_diag.data() );

    CHKERRXX( err );
}

matrix::builder::builder(
        matrix::size_type rows,
        matrix::size_type cols,
        const std::vector< matrix::size_type >& nnz_on_diag,
        const std::vector< matrix::size_type >& nnz_off_diag )
    : matrix( sized_matrix( rows, cols ) )
{

    assert( nnz_on_diag.size() == nnz_off_diag.size() );

    /* nnz_per_row[ i ] = nnz_on_diag[ i ] + nnz_off_diag[ i ]; */
    std::vector< matrix::size_type > nnz_per_row( nnz_on_diag.size() );
    std::transform( nnz_on_diag.begin(), nnz_on_diag.end(),
            nnz_off_diag.begin(), nnz_per_row.begin(),
            std::plus< matrix::size_type >() );

    /*
     * We calculate the sequential version's nonzero-per-row by simply
     * zip-summing the on- and off-diag vectors
     */
    auto err = MatSeqAIJSetPreallocation( this->ptr(),
            /*ignored*/ 0,
            nnz_per_row.data() );
    CHKERRXX( err );

    /* Set only the local values to reduce communication */
    matrix::size_type offset;
    MatGetOwnershipRange( this->ptr(), &offset, NULL );

    err = MatMPIAIJSetPreallocation( this->ptr(),
            /* ignored */ 0, nnz_on_diag.data() + offset,
            /* ignored */ 0, nnz_off_diag.data() + offset );

    CHKERRXX( err );
}

matrix::builder::builder( const matrix::builder& x ) : matrix( x.commit() ) {}

matrix::builder& matrix::builder::insert(
        matrix::size_type row,
        matrix::size_type col,
        matrix::scalar value ) {

    const size_type rows[] = { row };
    const size_type cols[] = { col };
    const scalar vals[] = { value };

    auto err = MatSetValues( this->ptr(),
            1, rows,
            1, cols,
            vals, INSERT_VALUES ); CHKERRXX( err );

    return *this;
}

matrix::builder& matrix::builder::insert(
        const std::vector< matrix::scalar >& nonzeros,
        const std::vector< matrix::size_type >& row_indices,
        const std::vector< matrix::size_type >& col_indices ) {

    assert( nonzeros.size() == col_indices.size() );

    for( unsigned int i = 0; i < row_indices.size() - 1; ++i ) {
        /* the difference between two consecutive elements are the indices in
         * the col&nonzero vectors where the current row i is stored
         */
        const auto row_entries = row_indices[ i + 1 ] - row_indices[ i ];

        /* empty row - skip ahead */
        if( !row_entries ) continue;

        /* MatSetValues takes an array */
        const matrix::size_type row_index[] = { matrix::size_type( i ) };

        /* offset to start insertion from */
        const auto offset = row_indices[ i ];

        const auto err = MatSetValues( this->ptr(),
                1, row_index,
                row_entries, col_indices.data() + offset,
                nonzeros.data() + offset, INSERT_VALUES );

        CHKERRXX( err );
    }

    return *this;
}

matrix::builder& matrix::builder::insert_row(
        matrix::size_type row,
        const std::vector< matrix::scalar >& values,
        matrix::size_type begin ) {

    const auto indices = range( begin, size_type( begin + values.size() ) );

    return this->insert_row( row, indices, values );
}

matrix::builder& matrix::builder::insert_row(
        matrix::size_type row,
        const std::vector< matrix::size_type >& indices,
        const std::vector< matrix::scalar >& values ) {

    /* MatSetValues takes an array */
    const matrix::size_type row_index[] = { row };

    const auto err = MatSetValues( this->ptr(),
            1, row_index,
            indices.size(), indices.data(),
            values.data(), INSERT_VALUES );

    CHKERRXX( err );

    return *this;
}

matrix matrix::builder::commit() const {
    /*
     * This is a peculiar case. So commit (and copy) doesn't actually -change-
     * the builder, but it does require mutability. the assemble method calls
     * MatAssemble from PETSc which flushes the caches and sets up the matrix
     * structure. However, this is an implementation detail, and the object
     * that the builder exposes is oblivious to this. We use the const_cast
     * trick to -appear- immutable, while we need mutability to flush our
     * caches.
     *
     * This is ok because the -visible- object does not change, and is for all
     * intents and purposes still const
     */

    const_cast< builder& >( *this ).assemble();
    /* Ensure we call matrix( const matrix& ). If we call matrix( const
     * builder& ) we would end up in a loop, because the matrix( builder& )
     * copy constructor must also ensure that commit is called.
     */
    return matrix( static_cast< const matrix& >( *this ) );
}

matrix::builder& matrix::builder::add(
        matrix::size_type row,
        matrix::size_type col,
        matrix::scalar value ) {

    const size_type rows[] = { row };
    const size_type cols[] = { col };
    const scalar vals[] = { value };

    auto err = MatSetValues( this->ptr(),
            1, rows,
            1, cols,
            vals, ADD_VALUES ); CHKERRXX( err );

    return *this;
}



matrix::builder& matrix::builder::assemble() {
    auto err = MatAssemblyBegin( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );
    err = MatAssemblyEnd( this->ptr(), MAT_FINAL_ASSEMBLY );
    CHKERRXX( err );

    return *this;
}

}
}
