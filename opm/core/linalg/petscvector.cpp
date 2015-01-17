#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscvector.hpp>

#include <cassert>
#include <memory>
#include <utility>

#include <iostream>

/* Slight hack to get things working. This sets the communicator used when
 * constructing new vectors - however, this should be configurable. When full
 * petsc support is figured out this will be changed.
 */
static inline MPI_Comm get_comm() { return PETSC_COMM_WORLD; }

namespace Opm {
namespace petsc {

/* ideally replace this with C++11's std::iota */
#ifndef HAVE_STD_IOTA
template< class ForwardIterator, class T >
static inline void iota( ForwardIterator first, ForwardIterator last, T value ) {
        while(first != last) *first++ = value++;
}
#endif //HAVE_STD_IOTA

template< typename T = vector::size_type, typename U = T >
static inline std::vector< T > range( U begin, U end ) {
    using namespace std;

    std::vector< T > x( end - begin );
    /* if std::iota is available, the function defined above shouldn't be
     * compiled in and std::iota should be used
     */
    iota( x.begin(), x.end(), begin );
    return x;
}

vector::vector() {
    /* Meant to be used by other constructors. */
    Vec x;
    VecCreate( get_comm(), &x );
    VecSetFromOptions( x );

    this->v.reset( x );
}

vector::vector( Vec x ) : v( x ) {}

vector::vector( const vector& x ) {
    Vec y;
    auto err = VecDuplicate( x, &y ); CHKERRXX( err );
    err = VecCopy( x, y ); CHKERRXX( err );

    this->v.reset( y );
}

vector::vector( vector::size_type size ) {
    vector vec;

    this->v.swap( vec.v );

    auto err = VecSetSizes( this->ptr(), PETSC_DECIDE, size );
    CHKERRXX( err );
}

vector::vector( vector::size_type size, vector::scalar x ) {
    vector vec( size );
    this->v.swap( vec.v );
    this->assign( x );
}

vector::vector( const std::vector< vector::scalar >& values ) {
    vector vec( values.data(), values.size() );
    this->v.swap( vec.v );
}

vector::vector( const std::vector< vector::scalar >& values,
                const std::vector< vector::size_type >& indexset ) {

    assert( indexset.size() == values.size() );

    vector vec( values.data(), indexset.data(), values.size() );
    this->v.swap( vec.v );
}

vector::vector( const vector::scalar* values, vector::size_type size ) {
    auto indices = range( 0, size );
    vector vec( values, indices.data(), size );

    this->v.swap( vec.v );
}

vector::vector(
        const vector::scalar* values,
        const vector::size_type* indexset,
        vector::size_type size ) {

    vector vec( size );

    this->v.swap( vec.v );
    this->set( values, indexset, size );
}

vector::operator Vec() const {
    return this->ptr();
}

vector::size_type vector::size() const {
    PetscInt x;
    auto err = VecGetSize( this->ptr(), &x ); CHKERRXX( err );
    return x;
}

void vector::assign( vector::scalar x ) {
    auto err = VecSet( this->ptr(), x ); CHKERRXX( err );
}

vector& vector::operator+=( vector::scalar rhs ) {
    auto err = VecShift( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

vector& vector::operator-=( vector::scalar rhs ) {
    return *this += -rhs;
}

vector& vector::operator*=( vector::scalar rhs ) {
    auto err = VecScale( this->ptr(), rhs ); CHKERRXX( err );
    return *this;
}

vector& vector::operator/=( vector::scalar rhs ) {
    return *this *= ( 1 / rhs );
}

vector& vector::operator+=( const vector& rhs ) {
    /* VecAXPY breaks if the vectors are not different. If they are equal,
     * *this += *this, then it is equal to *this *= 2.
     */
    if( this->ptr() == rhs.ptr() ) return *this *= 2;

    auto err = VecAXPY( this->ptr(), 1, rhs ); CHKERRXX( err );
    return *this;
}

vector& vector::operator-=( const vector& rhs ) {
    /* VecAXPY breaks if the vectors are not different. In the case of
     * *this -= *this, this is identical to assign( 0 ) or *this *= 0.
     */
    if( this->ptr() == rhs.ptr() ) return *this *= 0;

    auto err = VecAXPY( this->ptr(), -1, rhs ); CHKERRXX( err );
    return *this;
}

vector operator+( vector lhs, vector::scalar rhs ) {
    return lhs += rhs;
}

vector operator-( vector lhs, vector::scalar rhs ) {
    return lhs -= rhs;
}

vector operator*( vector lhs, vector::scalar rhs ) {
    return lhs *= rhs;
}

vector operator/( vector lhs, vector::scalar rhs ) {
    return lhs /= rhs;
}

vector operator+( vector lhs, const vector& rhs ) {
    return lhs += rhs;
}

vector operator-( vector lhs, const vector& rhs ) {
    return lhs -= rhs;
}

vector::scalar operator*( const vector& lhs, const vector& rhs ) {
    assert( lhs.size() == rhs.size() );

    return dot( lhs, rhs );
}

bool vector::operator==( const vector& other ) const {
    PetscBool eq;
    VecEqual( this->ptr(), other, &eq );
    return eq;
}

bool vector::operator!=( const vector& other ) const {
    return !( *this == other );
}

inline void vector::set( const vector::scalar* values,
        const vector::size_type* indices,
        vector::size_type size ) {

    auto err = VecSetValues( this->ptr(), size,
            indices, values, INSERT_VALUES ); CHKERRXX( err );

    err = VecAssemblyBegin( this->ptr() ); CHKERRXX( err );
    err = VecAssemblyEnd( this->ptr() ); CHKERRXX( err );
}

inline Vec vector::ptr() const {
    return this->v.get();
}

vector::scalar dot( const vector& lhs, const vector& rhs ) {
    vector::scalar x;
    VecDot( lhs, rhs, &x );
    return x;
}

vector::scalar sum( const vector& v ) {
    vector::scalar x;
    auto err = VecSum( v, &x ); CHKERRXX( err );
    return x;
}

vector::scalar max( const vector& v ) {
    vector::scalar x;
    auto err = VecMax( v, NULL, &x ); CHKERRXX( err );
    return x;
}

vector::scalar min( const vector& v ) {
    vector::scalar x;
    auto err = VecMin( v, NULL, &x ); CHKERRXX( err );
    return x;
}

}
}
