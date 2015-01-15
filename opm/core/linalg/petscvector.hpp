#ifndef OPM_PETSCVECTOR_H
#define OPM_PETSCVECTOR_H

/*
 * C++ bindings to petsc Vec.
 *
 * Provides an easy-to-use C++-esque implementation with no extra indirection
 * for petsc Vec. Slightly radical in design, but in spirit with petsc: no
 * direct memory access is allowed, and no iterators are provided. Due to the
 * nature of petsc there usually is no guarantee that the memory you want to
 * access lives in your process. All interactions with the vector should happen
 * through functions - possibly general higher-order functions at a later time.
 *
 * The idea is to get (some) of petsc's power, but easier to use. Tested to
 * without effort be able to use MPI. Supports arithmetic operators and should
 * be easy to reason about. Lifetime is managed through unique_ptr for safety
 * and simplicity - no need to implement custom assignment and move operators.
 *
 * Note that a default-constructed vector is NOT allowed. While this makes it
 * slightly harder to use in a class (you most likely shouldn't anyways), it
 * disallows more invalid states at compile-time.
 *
 * Also note that there are no set-methods for values. This is by design, but
 * they might be added later if there is a need for it. Structurally, this
 * leaves the vector immutable.
 *
 * Please note that all methods in the vector class only deal with it's
 * structure. This is by design - everything needed to compute something about
 * the vector is or will be provided as free functions, in order to keep
 * responsibilities separate.
 *
 * The vector gives no guarantees for internal representation.
 *
 * TODO: When full C++11 support can be used, most constructors can be
 * rewritten to use ineheriting constructors.
 */

#include <opm/core/linalg/petsc.hpp>

#include <petscvec.h>

#include <memory>
#include <vector>

namespace Opm {
namespace petsc {

    class vector {
        public:
            typedef PetscScalar scalar;
            typedef PetscInt size_type;

            /* Just take ownership over a given Vec. */
            explicit vector( Vec );
            vector( const vector& );

            explicit vector( size_type );
            explicit vector( size_type, scalar );

            vector( const std::vector< scalar >& );
            vector( const std::vector< scalar >&,
                    const std::vector< size_type >& indexset );

            vector( const scalar*, size_type );
            vector( const scalar*, const size_type* indexset, size_type );

            /* implicit conversion to Vec.
             *
             * We want this so the class can be dropped into petsc functions.
             */
            operator Vec() const;

            size_type size() const;

            /* assign a value to -all- cells */
            void assign( scalar );

            vector& operator+=( scalar );
            vector& operator-=( scalar );
            vector& operator*=( scalar );
            vector& operator/=( scalar );

            /* these currently do not have to be friends (since vector is
             * implicitly convertible), but they are for least possible
             * surprise
             */
            friend vector operator+( vector, scalar );
            friend vector operator-( vector, scalar );
            friend vector operator*( vector, scalar );
            friend vector operator/( vector, scalar );

            friend scalar operator*( const vector&, const vector& );

            bool operator==( const vector& ) const;
            bool operator!=( const vector& ) const;

        private:
            vector();

            /* We rely on unique_ptr to manage lifetime semantics. */
            struct del { void operator()( Vec v ) { VecDestroy( &v ); } };
            std::unique_ptr< _p_Vec, del > v;

            inline void set( const scalar*, const size_type*, size_type );

            /* a convenient, explicit way to get the underlying Vec */
            inline Vec ptr() const;
    };

    vector::scalar dot( const vector&, const vector& );

    vector::scalar sum( const vector& );
    vector::scalar max( const vector& );
    vector::scalar min( const vector& );

}
}

#endif //OPM_PETSCVECTOR_H
