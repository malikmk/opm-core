#ifndef OPM_PETSC_MIXINS_H
#define OPM_PETSC_MIXINS_H

#include <memory>

namespace Opm {
namespace petsc {

/*
 * Deleters are implemented in the same files that declares the class they're
 * to delete, i.e. petscvector.hpp for vector deleter.
 */
template< typename T > struct deleter;

/*
 * petsc typedefs its handles from _p_Vec* (etc) to Vec. It is certainly nicer
 * to, in the class declaration, declare the type you -actually- want exposed,
 * so we use a template trick to remove the pointer part of the type and infer
 * the underlying type. This is done by first declaring a shell template class,
 * to introduce the symbol, then specialise for a pointer type. Notice that
 * uptr is only forward declared and never gets an implementation - this is by
 * design, so that we get compiler errors in case it is used wrong.
 */
template< typename T > class uptr;

/*
 * Here we take the pointer typedef, e.g. Vec and Mat, and strip the pointer so
 * that unique_ptr can use them. This avoids double indirection, making sure
 * the uptr mixin provides no extra overhead.
 */
template< typename T >
class uptr< T* > : private std::unique_ptr< T, deleter< T > > {
    private:
        typedef T* pointer;
        typedef std::unique_ptr< T, deleter< T > > base;

    public:
        /* We want the same constructors as in unique_ptr,
         * but generally not the other methods. The reasoning here is that
         * avoiding direct pointer access makes code easier to change in the
         * future and less reliant on actual implementation.
         */
        using base::unique_ptr;
        using base::operator=;

        operator pointer() const;

    protected:
        inline pointer ptr() const;
};

/*
 * Implementations
 */

template< typename T >
uptr< T* >::operator uptr< T* >::pointer() const {
    return this->ptr();
}

template< typename T >
T* uptr< T* >::ptr() const {
    return this->get();
}

}
}

#endif //OPM_PETSC_MIXINS_H
