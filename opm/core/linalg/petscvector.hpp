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

    /// Class implementing C++-support for PETSc's Vec object. Provides no
    /// extra indirection and can be used as a handle for vanilla PETSc
    /// functions, should the need be.

    class vector {
        public:
            typedef PetscScalar scalar;
            typedef PetscInt size_type;

            /// Takes ownership of a Vec produced by some other means. The
            /// destruction and lifetime is now managed by vector
            explicit vector( Vec );
            /// Copy constructor
            vector( const vector& );

            /// Constructor. Does not populate the vector with values.
            /// \param[in] size     Number of elements
            explicit vector( size_type );
            /// Constructor. Populate [0..n-1] with scalar. This is equivalent
            /// vector v( size ); v.assign( scalar );
            /// \param[in]  size    Number of elements
            /// \param[in]  scalar  Value to set
            explicit vector( size_type, scalar );

            /// Construct from std::vector.
            /// \param[in] vector   std::vector to copy from
            vector( const std::vector< scalar >& );
            /// Construct from std::vector at the indices provided by
            /// indexset. Negative indices are ignored.
            /// \param[in] vector   std::vector to copy from
            /// \param[in] indexset indices to assign values from the vector.
            vector( const std::vector< scalar >&,
                    const std::vector< size_type >& indexset );

            /// Construct from array. Provided to support legacy APIs and
            /// C or C-style libraries. Works like the std::vector-constructor,
            /// but with raw arrays.
            /// \param[in] vector   array to copy from
            /// \param[in] size     size of the array
            vector( const scalar*, size_type );
            /// Construct from array at the indices provided by indexset.
            /// Provided to support legacy APIs and C or C-style libraries.
            /// Works like the std::vector-constructor, but with raw arrays.
            /// \param[in] vector   array to copy from
            /// \param[in] indexset indices to assign values from the vector.
            /// \param[in] size     size of the array
            vector( const scalar*, const size_type* indexset, size_type );

            /// Implicit conversion to Vec for compability with PETSc functions
            operator Vec() const;

            /// Get vector size.
            /// \param[out] size    Number of elements in the vector
            size_type size() const;

            /// Assign a value to all elements in the vector.
            /// \param[in] scalar  The value all elements will have
            void assign( scalar );

            /// Add a value to all elements in the vector
            /// \param[in] scalar   Value to add to all elements
            vector& operator+=( scalar );
            /// Subtract a value from all elements in the vector
            /// \param[in] scalar   Value to subtract from all elements
            vector& operator-=( scalar );
            /// Scalar multiplication
            /// \param[in] scalar   Value to scale with
            vector& operator*=( scalar );
            /// Inverse scalar multiplication (division)
            /// \param[in] scalar   Value to scale with
            vector& operator/=( scalar );

            /// Vector addition, X + Y. Induces a copy if X == Y
            /// \param[in] vector   Vector to add
            vector& operator+=( const vector& );
            /// Vector subtraction, X - Y. Induces a copy if X == Y
            /// \param[in] vector   Vector to subtract
            vector& operator-=( const vector& );

            /* these currently do not have to be friends (since vector is
             * implicitly convertible), but they are for least possible
             * surprise
             */
            friend vector operator+( vector, scalar );
            friend vector operator-( vector, scalar );
            friend vector operator*( vector, scalar );
            friend vector operator/( vector, scalar );

            friend vector operator+( vector, const vector& );
            friend vector operator-( vector, const vector& );
            friend scalar operator*( const vector&, const vector& );

            /// Equality check
            /// \param[out] eq  Returns true if equal, false if not
            bool operator==( const vector& ) const;
            /// Unequality check
            /// \param[out] eq  Returns false if equal, true if not
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

    /// Calculate the dot product of two vectors
    vector::scalar dot( const vector&, const vector& );

    /// Calculate the sum of all values in the vector
    vector::scalar sum( const vector& );
    /// Find the biggest element in the vector
    vector::scalar max( const vector& );
    /// Find the smallest element in the vector
    vector::scalar min( const vector& );

}
}

#endif //OPM_PETSCVECTOR_H
