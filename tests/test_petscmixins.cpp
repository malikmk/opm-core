#include <config.h>

#include <opm/core/linalg/petsc.hpp>
#include <opm/core/linalg/petscmixins.hpp>

#if HAVE_DYNAMIC_BOOST_TEST
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE MatrixTest
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
namespace petsc {
template<> struct deleter< int > : std::default_delete< int > {};
}
}


struct unique : public Opm::petsc::uptr< int* > {
    typedef Opm::petsc::uptr< int* > base;
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
