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

static inline std::vector< vector::size_type >
range( vector::size_type begin, vector::size_type end ) {
    using namespace std;

    std::vector< vector::size_type > x( end - begin );
    /* if std::iota is available, the function defined above shouldn't be
     * compiled in and std::iota should be used
     */
    iota( x.begin(), x.end(), begin );
    return x;
}

static inline Vec default_vector() {
    /* Meant to be used by other constructors. */
    Vec x;
    VecCreate( get_comm(), &x );
    VecSetFromOptions( x );

    return x;
}

/* To provide exception safety, wrap the vector handles in a manged
 * (unique_ptr) object. While it looks a bit ugly, it is used basically only in
 * a few functions, so it should be ok.
 *
 * Using delegating constructors (gcc4.7-4.8) would be better, so a future TODO
 * is to remove these functions in favour of delegating constructors.
 */

static inline Vec copy_vector( Vec x ) {
    Vec y;
    auto err = VecDuplicate( x, &y ); CHKERRXX( err );
    std::unique_ptr< _p_Vec, deleter< _p_Vec > > result( y );
    err = VecCopy( x, y ); CHKERRXX( err );

    return result.release();
}

static inline Vec sized_vector( vector::size_type size ) {
    std::unique_ptr< _p_Vec, deleter< _p_Vec > > result( default_vector() );

    auto err = VecSetSizes( result.get(), PETSC_DECIDE, size );
    CHKERRXX( err );

    return result.release();
}

static inline Vec set_vector( const std::vector< vector::scalar >& values,
                            const std::vector< vector::size_type >& indices ) {

    assert( values.size() == indices.size() );

    std::unique_ptr< _p_Vec, deleter< _p_Vec > >
        result( sized_vector( values.size() ) );

    auto err = VecSetValues( result.get(), values.size(),
            indices.data(), values.data(),
            INSERT_VALUES );
    CHKERRXX( err );

    err = VecAssemblyBegin( result.get() ); CHKERRXX( err );
    err = VecAssemblyEnd( result.get() ); CHKERRXX( err );

    return result.release();

}

vector::vector( const vector& x ) : uptr( copy_vector( x ) ) {}

/*
 * When we can move to full C++11 support, these should ideally be implemented
 * with delegating constructors.
 */

vector::vector( vector::size_type size ) : uptr( sized_vector( size ) ) {}

vector::vector( vector::size_type size, vector::scalar x ) :
        uptr( sized_vector( size ) )
{
    this->assign( x );
}

vector::vector( const std::vector< vector::scalar >& values ) :
    uptr( set_vector( values, range( 0, values.size() ) ) )
{}

vector::vector( const std::vector< vector::scalar >& values,
                const std::vector< vector::size_type >& indexset ) :
    uptr( set_vector( values, indexset ) )
{}

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
    /* VecAXPY breaks if the vectors are not different, so if they are, copy
     * one and use that for the addition
     */
    if( this->ptr() == rhs.ptr() ) {
        auto rhs_copy = rhs;
        return *this += rhs_copy;
    }

    auto err = VecAXPY( this->ptr(), 1, rhs ); CHKERRXX( err );
    return *this;
}

vector& vector::operator-=( const vector& rhs ) {
    /* VecAXPY breaks if the vectors are not different, so if they are, copy
     * one and use that for the addition
     */
    if( this->ptr() == rhs.ptr() ) {
        auto rhs_copy = rhs;
        return *this -= rhs_copy;
    }

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

    /*
     * TODO: reimplement this smarter, i.e. use setvalues and local updates if
     * possible
     */

    auto err = VecSetValues( this->ptr(), size,
            indices, values, INSERT_VALUES ); CHKERRXX( err );

    err = VecAssemblyBegin( this->ptr() ); CHKERRXX( err );
    err = VecAssemblyEnd( this->ptr() ); CHKERRXX( err );
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
