#include <config.h>

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmixins.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE PetscMixinTest
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

/*
 * This file tests the correctness of the mixins. The benefit of doing this is
 * that we can cleany verify the correctness of the semantics and operations
 * provided by the mixins, without dealing with the details of specific
 * classes. This means tests become smaller, and it will be obsolete to test
 * that, say, the petsc::vector assignment operators works as intended.
 */

namespace Opm {
namespace Petsc {
template<> struct deleter< int > : std::default_delete< int > {};
}
}


struct unique : public Opm::Petsc::uptr< int* > {
    typedef Opm::Petsc::uptr< int* > base;
    using base::operator=;

    unique( int* x ) : base( x ) {}
    unique( unique&& x ) : base( std::move( x ) ) {}
    unique( const unique& other ) : base( new int( *other.get() ) ) {}

    const int* get() const { return this->ptr(); }
};

static unique capture( int* p ) {
    return unique( p );
}

BOOST_AUTO_TEST_CASE( assignment ) {

    int* p = new int( 3 );

    unique orig( p );
    /* Sanity check - if this fails, simply taking ownership of a pointer, none
     * of the other tests can be verified
     */
    BOOST_CHECK( p == orig.get() );

    /* Test move assignment operator from std::move */
    /* This should call the move constructor (of moved) and make so that moved
     * now is the owner of p. This can fail in two ways:
     * #1: moved's internal pointer differs from the one it is supposed to get
     * transferred(p)
     * #2: both orig and moved holds ownership over p. If so, you should see a
     * double free
     */
    unique moved = std::move( orig );
    BOOST_CHECK( p );
    BOOST_CHECK( p == moved.get() );

    /* Copy the contents */
    auto copied = moved;

    /* These should now NOT be similar, i.e. copied should have a totally new
     * pointer, but their -value- should remain equal
     */
    BOOST_CHECK( moved.get() != copied.get() );
    BOOST_CHECK( *moved.get() == *copied.get() );

    int* cap = new int( 5 );
    /* Make sure move constructor is correctly called when returning the
     * object by value
     */
    unique captured = capture( cap );
    BOOST_CHECK( captured.get() == cap );
}

class ebc_test : public Opm::Petsc::explicit_bool_conversion< ebc_test > {};
bool explicit_boolean_test( const ebc_test& ) {
    return true;
}

BOOST_AUTO_TEST_CASE( explicit_bool_conversion_operator ) {
    /*
     * Since we do not yet have access to the C++11 explicit conversion
     * operator keyword, we emulate it through the so-called safe bool idiom.
     *
     *  See: http://en.wikibooks.org/wiki/More_C++_Idioms/Safe_bool
     *
     *  This code tests the mixin and makes sure it behaves as expected.
     *  Relies on CRTP, so every bool conversion class must implement the
     *  following function overload: explicit_boolean_test( const class& )
     *
     *  Adding an automatic test for the case that is NOT supposed to compile
     *  is unfortunately rather hard, so it is currently not covered by this
     *  automatic test case.
     */

    ebc_test ebc;
    BOOST_CHECK( ebc );
    BOOST_CHECK( ebc == true );
    BOOST_CHECK( ebc != false );

    /*
     * If this is uncommented, the test should not compile.
     *
     * ebc_test ebc1;
     * int x = 0;
     * BOOST_CHECK( ebc == x );
     * BOOST_CHECK( ebc1 == ebc );
     */
}

enum class returned_from { newtype, rawtype };

struct age : public Opm::Petsc::newtype< int > {
    /*
     * Pre-C++11 support (using keyword works for GCC >= 4.8)
     */
    template< typename T >
    age( T&& t ) : Opm::Petsc::newtype< int >(
            std::forward< T >( t ) ) {}
    /*
     * C++11-version:
     * using Opm::petsc::newtype< int >::newtype;
     */
};

// Test the mknewtype macro too
mknewtype( height, int );

returned_from test_newtype( age x ) {
    return returned_from::newtype;
}

returned_from test_newtype( height x ) {
    return returned_from::newtype;
}

returned_from test_newtype( int x ) {
    return returned_from::rawtype;
}

BOOST_AUTO_TEST_CASE( newtypes ) {
    int integer = 0;
    age x = 1;
    age y( integer );
    height z = 2;

    BOOST_CHECK( test_newtype( integer ) == returned_from::rawtype );
    BOOST_CHECK( test_newtype( x ) == returned_from::newtype );
    BOOST_CHECK( test_newtype( y ) == returned_from::newtype );
    BOOST_CHECK( test_newtype( z ) == returned_from::newtype );
    BOOST_CHECK( test_newtype( height( x ) ) == returned_from::newtype );
}
