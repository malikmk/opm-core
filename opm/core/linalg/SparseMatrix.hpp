#ifndef OPM_SPARSEMATRIX_HEADER
#define OPM_SPARSEMATRIX_HEADER

#include <memory>
#include <vector>

/*
 * An interface for sparse matrices, see the announcement:
 * http://www.opm-project.org/pipermail/opm/2014-October/000664.html
 *
 * Motivation for this file and its contents:
 * We would like to support multiple linear solver backends. A lot of these
 * solvers uses their own data types, and in some cases code in the OPM project
 * is very backend specific. In an attempt to resolve this I introduce a
 * unified sparse matrix interface that ideally should be compatible with all
 * solver libraries.
 *
 * It is designed in such a way that we can utilize "native types" (i.e. data
 * structures provided by a specific solver library). This is achieved through
 * template specialization and composition. This header file provides an
 * interface you can assume are compatible with any underlying container for
 * various linear solver libraries. It also provides a default implementation
 * using our own CSR Matrix representation, using a simple C-compatible struct
 * as actual storage.
 */

namespace Opm {

    namespace unmanaged {
        /* the plain, C-compatible CSR matrix structure.
         *
         * rows_ gives the number of rows in the matrix, but rows + 1 places
         * are allocated.
         *
         * each row's indices and values are given by the difference between
         * row_ind_[ i + 1 ] and row_ind_[ i ]. This access is always safe
         * because of the extra value set. The final value of row_ind_should
         * always be equal to nonzeros_
         */
        template< typename Scalar = double >
        struct CSR {
            CSR() = default;
            CSR( std::size_t rows, std::size_t nonzeros, Scalar* values, int*, int* );
            std::size_t rows_;
            std::size_t nonzeros_;
            Scalar* values_;
            int* col_ind_;
            int* row_ind_;
        };
    }

    template< typename Scalar = double >
    struct CSR : public unmanaged::CSR< Scalar >{

        /* iterators are templated on T in order to support const_iterators */
        template< typename T, typename Derived >
        class base_itr {
            /* A CRTP mixin to implement trivial variations of common iterator
             * operations. Assumes that the derived class implements
             * ++operator, operator== and operator*
             */
            public:
                Derived operator++( int );
                Derived operator--( int );
                bool operator!=( const Derived& other ) const;
                T* operator->();
        };

        template< typename T >
        class col_itr : public base_itr< T, col_itr< T > > {
            public:

                /* forward iterator for the columns of a row. Reuses
                 * base_itr for implementation of some operations, but is
                 * pretty straight forward.
                 */

                col_itr() = default;
                col_itr( Scalar* val, int* col, std::size_t ind );

                col_itr& operator++();
                col_itr& operator--();
                bool operator==( const col_itr& ) const;

                T& operator*();

                std::size_t index() const;

            private:
                Scalar* values_;
                int* col_ind_;
                std::size_t index_;
        };

        template< typename T >
        class row_ref {
            /*
             * References some row in the matrix. This is a proxy class to
             * access some operations and create iterators for the matrix, e.g.
             * operator[].
             *
             * It is also intented for use as a mixin in row_itr, but this is
             * just a (nifty) implementation detail. It is not meant to be
             * created directly from user code, but rather be (correctly)
             * constructed through iterators or operator[].
             */
            public:
                col_itr< T > begin();
                col_itr< T > end();

                row_ref() = default;

                /* undefined behaviour if given matrix position is zero
                 *
                 * This does NOT guarantee constant-time lookup.
                 */
                T& operator[]( std::size_t );
                const T& operator[]( std::size_t ) const;

                std::size_t index() const;
                std::size_t size() const;

            protected:
                /* this class is not meant to be constructed directly */
                row_ref( Scalar* val, int* col, int* row, std::size_t index );

                Scalar* values_;
                int* col_ind_;
                int* row_ind_;
                std::size_t index_;
        };

        template< typename T >
        class row_itr : private row_ref< T >, public base_itr< T, row_itr< T > > {
            /* Bi-directional row iterator. Behaves as expected, but uses some
             * C++ tricks:
             *
             * * CRTP mixin (base_itr) for a shorter implementation.
             * * privately inherits from row_ref and (re)uses its fields for
             * its data. Constructor is basically a forward call to that. This
             * composition allows us to reuse its data fields (which would've
             * been private anyways). 
             *
             * The real goal of the private implementation, however, was a
             * zero-copy dereference of a row_ref. operator* an operator-> now
             * simply returns a ref to -the instance itself-, now
             * re-interpreted as a row_ref. This gives iterator access when
             * used as a row_itr and  row_ref access when accessed as that.
             * This, however, is just an implementation detail and not really
             * visible from user code.
             */
            public:
                row_itr( Scalar* val, int* col, int* row, std::size_t index );

                row_itr& operator++();
                row_itr& operator--();
                bool operator==( const row_itr& ) const;

                row_ref< T >& operator*();
                /* overrides base_itr::operator-> */
                row_ref< T >* operator->();

                /* for consistency with col_itr (and compability with Dune's
                 * BCRSMatrix iterator, index() is also available as a
                 * method on the iterator
                 */
                using row_ref< T >::index;
        };

        typedef Scalar type;
        typedef Scalar scalar;

        typedef typename unmanaged::CSR< Scalar > unmanaged;

        typedef Scalar& Reference;
        typedef Scalar* Pointer;
        typedef row_itr< Scalar > iterator;
        typedef row_itr< const Scalar > const_iterator;

        /* no copy or assignment construction, behave like a unique_ptr */
        CSR( const CSR& ) = delete;
        CSR& operator=( const CSR& ) = delete;
        CSR& operator=( CSR&& ) = default;

        CSR() = default;
        CSR( std::size_t rows, std::size_t nonzeros, Scalar* values, int*, int* );
        CSR( CSR&& );
        ~CSR();

        /*
         * mimics std::unique_ptr. get just returns an unmanaged copy of the
         * object, whereas release releases (duh) ownership. If you call
         * release, make sure to free the resources afterwards.
         */
        unmanaged get();
        unmanaged release();
    };

    template< typename Scalar = double, typename Storage = CSR< Scalar > >
    class SparseMatrix : private Storage {
        public:
            typedef Scalar type;
            typedef Scalar scalar;

            typedef typename Storage::unmanaged unmanaged;

            typedef Scalar& Reference;
            typedef Scalar* Pointer;
            typedef typename Storage::iterator iterator;
            typedef typename Storage::const_iterator const_iterator;

        SparseMatrix() = default;
        SparseMatrix( SparseMatrix&& ) = default ;
        SparseMatrix& operator=( SparseMatrix&& );

        SparseMatrix( const SparseMatrix& ) = delete;
        SparseMatrix& operator=( const SparseMatrix& ) = delete;

        // This constructor takes -ownership- over the given data
        // TODO: communicate this through interface. Take unique_ptr?
        SparseMatrix( std::size_t rows, std::size_t nonzero,
                Scalar* values, int* col_ind, int* row_ind );

        /*
         * A test to see if the (x,y) coordinate is nonzero. log(n) complexity
         * for CSR, typically does not give constant-time guarantees.
         */
        bool exists( std::size_t, std::size_t ) const;

        /* bounds checked operator[][] (Not implemented yet) */
        Scalar& at( std::size_t, std::size_t );
        const Scalar& at( std::size_t, std::size_t ) const;

        typename Storage::template row_ref< Scalar > operator[]( std::size_t );
        typename Storage::template row_ref< const Scalar > operator[]( std::size_t ) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        std::size_t rows() const;
        /*
         * cols() does -not- actually guarantee returning the correct result,
         * at least for CSR storage. The reason for this is that there could be
         * columns all filled with zeros which aren't represented (and really
         * not very useful) that are supposed to be there, but aren't. For most
         * cases this should be irrelevant, and this estimate is ok.
         *
         * O(n) with the size of nonzeros.
         */
        std::size_t cols() const;
        std::size_t nonzeros() const;

        /*
         * Get the underlying structure. Release also releases ownership
         */
        typename Storage::unmanaged get();
        typename Storage::unmanaged release();
    };

    template< typename Scalar = double >
    class SparseMatrixBuilder {
        public:
            void add( std::size_t i, std::size_t j, Scalar val );
            SparseMatrix< Scalar, CSR< Scalar > > to_csr();

        private:
            struct COO {
                std::size_t row_;
                std::size_t col_;
                Scalar value_;

                bool operator==( const COO& o ) const {
                    return this->row_ == o.row_
                        && this->col_ == o.col_;
                }

                bool operator<( const COO& o ) const {
                    if( this->row_ > o.row_ ) return false;
                    return this->row_ < o.row_
                        || this->col_ < o.col_;
                }
            };

            bool sorted_;
            std::vector< COO > values_;
    };
}

#include "SparseMatrix_impl.hpp"

#endif //OPM_SPARSEMATRIX_HEADER
