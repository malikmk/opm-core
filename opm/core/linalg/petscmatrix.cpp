#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>
#include <opm/core/linalg/petscmatrix.hpp>

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

/* Slight hack to get things working. This sets the communicator used when
 * constructing new vectors - however, this should be configurable. When full
 * petsc support is figured out this will be changed.
 */
static inline MPI_Comm get_comm() { return PETSC_COMM_WORLD; }

/* ideally replace this with C++11's std::iota */
#ifndef HAVE_STD_IOTA
template< class ForwardIterator, class T >
static inline void iota( ForwardIterator first, ForwardIterator last, T value ) {
        while(first != last) *first++ = value++;
}
#endif //HAVE_STD_IOTA

template< typename T = Opm::petsc::matrix::size_type, typename U >
static inline std::vector< T > range( U begin, U end ) {
    using namespace std;

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

matrix::matrix() {
    /* Meant to be used by other constructors. */
    Mat x;
    MatCreate( get_comm(), &x );
    MatSetFromOptions( x );

    this->m.reset( x );
}

matrix::matrix( Mat x ) : m( x ) {}

matrix::matrix( const matrix& x ) {
    Mat y;
    auto err = MatDuplicate( x, MAT_COPY_VALUES, &y ); CHKERRXX( err );
    err = MatCopy( x, y, SAME_NONZERO_PATTERN ); CHKERRXX( err );

    this->m.reset( y );
}

matrix::matrix( const matrix::builder& builder ) : m( builder.commit().m ) {}

matrix::matrix( matrix::builder&& builder ) :
    /* assemble and move the builder's matrix object into this one */
    m( builder.assemble().m.m.release() ) 
{}

matrix::matrix( matrix::size_type rows, matrix::size_type cols ) {
    matrix mat;

    this->m.swap( mat.m );

    auto err = MatSetSizes( this->ptr(),
            PETSC_DECIDE, PETSC_DECIDE,
            rows, cols );
    CHKERRXX( err );
}

matrix::matrix( const std::vector< matrix::scalar >& values,
                matrix::size_type rows,
                matrix::size_type cols ) {

    assert( values.size() == rows * cols );

    /* We know we want a dense matrix from this constructor, so we can afford
     * to assume MatCreateDense is the right call
     */
    Mat mat;
    auto err = MatCreateDense( get_comm(),
                        PETSC_DECIDE, PETSC_DECIDE,
                        rows, cols,
                        NULL, &mat );
    CHKERRXX( err );
    this->m.reset( mat );

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

matrix::operator Mat() const {
    return this->m.get();
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

inline Mat matrix::ptr() const {
    return this->m.get();
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
    matrix x( rhs );
    x.hermitian_transpose();
    return x;
}

matrix::builder::builder( matrix::size_type rows, matrix::size_type cols ) :
    m( matrix( rows, cols ) )
{
    auto err = MatSetUp( this->ptr() );
    CHKERRXX( err );
}

matrix::builder::builder( const matrix::builder& x ) : m( x ) {
    MatAssemblyBegin( this->m, MAT_FLUSH_ASSEMBLY );
    MatAssemblyEnd( this->m, MAT_FLUSH_ASSEMBLY );
}

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
    return const_cast< builder& >( *this ).assemble().m;
}

matrix matrix::builder::move() {
    /* 
     * We assume that matrix' constructor( builder&& ) handles the actual move
     * and assembly. The reason for this is that we want return builder; to
     * construct matrices, and we want the move constructor to behave nicely.
     * Furthermore, this leads to less code duplication.
     */
    return *this;
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

inline Mat matrix::builder::ptr() const {
    return this->m;
}

}
}
