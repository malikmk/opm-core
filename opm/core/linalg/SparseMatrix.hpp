#ifndef OPM_SPARSEMATRIX_HEADER
#define OPM_SPARSEMATRIX_HEADER

#include <memory>
#include <vector>
#include <map>

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
 * as actual storage. The default implementation is intended primarily as a
 * safe fallback and a reference. For high performance code you should use the
 * interface, but as relays to the native types your library provides.
 *
 * NOTICE: This is currently an experimental implementation and will likely
 * break in future revisions.
 *
 * NOTICE: This implementations uses a LOT of non-trivial C++ tricks to
 * accomplish its goals. I've tried to comment it whenever something might be
 * unclear, but noone is perfect. Let me know if something is missing:
 * mailto:jorgekva@stud.ntnu.no
 *
 * If you are extending this interface to a new backend, please also see the
 * tests in tests/test_sparsematrix.cpp.
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
    struct CSR : public unmanaged::CSR< Scalar > {

        /* iterators are templated on T in order to support const_iterators */
        template< typename T, typename Derived >
        class base_itr {
            /* A CRTP mixin to implement trivial variations of common iterator
             * operations. Assumes that the derived class implements
             * ++operator, operator== and operator*
             */
            public:
                base_itr() = default;

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
                row_itr() = default;
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

        typedef std::size_t size_type;

        typedef typename unmanaged::CSR< Scalar > unmanaged;

        typedef Scalar& Reference;
        typedef Scalar* Pointer;
        typedef row_itr< Scalar > iterator;
        typedef row_itr< const Scalar > const_iterator;

        /* no copy or assignment construction, behave like a unique_ptr */
        CSR( const CSR& ) = delete;
        CSR& operator=( const CSR& ) = delete;
        CSR& operator=( CSR&& ) = default;

        CSR();
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

    template< typename Storage = CSR< double > >
    class SparseMatrix : private Storage {
        public:
            typedef typename Storage::Scalar scalar;
            typedef scalar type;

            typedef std::size_t size_type;

            typedef typename Storage::unmanaged unmanaged;

            typedef typename Storage::iterator iterator;
            typedef typename Storage::const_iterator const_iterator;

            typedef typename Storage::template row_ref< scalar > row_ref;
            typedef typename Storage::template row_ref< const scalar > const_row_ref;

        SparseMatrix() = default;
        SparseMatrix( SparseMatrix&& ) = default;
        SparseMatrix& operator=( SparseMatrix&& );

        SparseMatrix( const SparseMatrix& ) = delete;
        SparseMatrix& operator=( const SparseMatrix& ) = delete;

        // This constructor takes -ownership- over the given data
        // TODO: communicate this through interface. Take unique_ptr?
        SparseMatrix( size_type rows, size_type nonzero,
                scalar* values, int* col_ind, int* row_ind );

        /*
         * A test to see if the (x,y) coordinate is nonzero. log(n) complexity
         * for CSR, typically does not give constant-time guarantees.
         */
        bool exists( size_type, size_type ) const;

        /* bounds checked operator[][] (Not implemented yet) */
        scalar& at( size_type, size_type );

        row_ref operator[]( size_type );
        const_row_ref operator[]( size_type ) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        size_type rows() const;
        /*
         * cols() does -not- actually guarantee returning the correct result,
         * at least for CSR storage. The reason for this is that there could be
         * columns all filled with zeros which aren't represented (and really
         * not very useful) that are supposed to be there, but aren't. For most
         * cases this should be irrelevant, and this estimate is ok.
         *
         * O(n) with the size of nonzeros.
         */
        size_type cols() const;
        size_type nonzeros() const;

        /*
         * Get the underlying structure. Allows you to directly access the
         * underlying object, at the cost of being unusable with other
         * implementations. The obvious use case for this is algorithm
         * development and testing, but it shouldn't be used unless absolutely
         * necessary in production code.
         */
        Storage& unsafe();
    };

    struct CO {
        CO( std::size_t r, std::size_t c ) : row_( r ), col_( c ) {}

        bool operator==( const CO& o ) const {
            return this->row_ == o.row_
                && this->col_ == o.col_;
        }

        bool operator<( const CO& o ) const {
            if( this->row_ > o.row_ ) return false;
            return this->row_ < o.row_
                || this->col_ < o.col_;
        }

        std::size_t row_;
        std::size_t col_;
    };

    template< typename S, typename T > class SparseMatrixBuilder;

    namespace {
        template< typename Scalar, typename Storage, typename Return >
        struct conv {
            static SparseMatrix< Return >
                convert( SparseMatrixBuilder< Scalar, Storage >* );
        };
    }

    template< typename Scalar = double, typename Storage = std::map< CO, Scalar > >
    class SparseMatrixBuilder : private Storage {
        public:
            SparseMatrixBuilder() = default;
            SparseMatrixBuilder( std::size_t rows, std::size_t cols );
            SparseMatrixBuilder( std::size_t rows, std::size_t cols, std::size_t nonzeros );

            void setSize( std::size_t rows, std::size_t ) {
                this->rows = rows;
            }

            void add( std::size_t i, std::size_t j, Scalar val );
            template< typename Return > SparseMatrix< Return > convert();

        private:
            int rows = 0;
            template< typename S, typename U, typename T >
            friend class conv;
    };
}

#include "SparseMatrix_impl.hpp"

#endif //OPM_SPARSEMATRIX_HEADER
