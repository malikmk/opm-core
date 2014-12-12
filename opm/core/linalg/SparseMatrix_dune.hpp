#if HAVE_DUNE_ISTL

#ifndef OPM_SPARSEMATRIX_DUNE
#define OPM_SPARSEMATRIX_DUNE

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>

namespace Opm {

    /*
     * For rationale and some elaboration, see SparseMatrix_csr.hpp
     */
    namespace {
    template<>
    template< typename Scalar, typename B >
    struct conv< Scalar, std::map< CO, Scalar >, Dune::BCRSMatrix< B > > {
    static SparseMatrix< Dune::BCRSMatrix< B > >
        convert( SparseMatrixBuilder< Scalar, std::map< CO, Scalar > >* obj ) {

            typedef Dune::BCRSMatrix< B > Mat;

            auto rows = obj->rows;
            Mat A( rows, rows, obj->size(), Mat::row_wise );

            auto valitr = obj->cbegin();
            for( auto row = A.createbegin(); row != A.createend(); ++row ) {
                const int ri = row.index();
                for( ; valitr->first.row_ == ri; ++valitr ) {
                    row.insert( valitr->first.col_ );
                }
            }

            for( auto itr = obj->cbegin(); itr != obj->cend(); ++itr ) {
                A[ itr->first.row_ ][ itr->first.col_ ] = itr->second;
            }

            return A;
        }
    };

    }

    /*
     * Dune::BCRSMatrix' interface happens to be very similar to
     * Opm::SparseMatrix. The entire implementation is simple forwards or
     * aliases.
     */

    template<>
    template< typename B >
    class SparseMatrix< Dune::BCRSMatrix< B > >
    : private Dune::BCRSMatrix< B > {
        private:
            typedef Dune::BCRSMatrix< B > Storage;

        public:
            typedef typename Storage::field_type scalar;
            typedef scalar type;

            typedef typename Storage::size_type size_type;

            typedef Storage unmanaged;

            typedef typename Storage::iterator iterator;
            typedef typename Storage::const_iterator const_iterator;

            /*
             * NOTICE:
             * operator[] does NOT return row_ref as in the (basic) interface,
             * but rather row_ref&. This is an implementation detail from DUNE.
             */
            typedef typename Storage::row_type row_ref;
            typedef const typename Storage::row_type const_row_ref;

        SparseMatrix() = default;
        /*
         * Not implemented yet. Relies on Dune move constructors
         * SparseMatrix( Storage&& );
         */

        SparseMatrix( const Storage& );

        /*
         * Not implemented yet.
         * Currently code depends on them. Will be sorted out along with move
         * semantics
         * SparseMatrix( const SparseMatrix& ) = delete;
         * SparseMatrix& operator=( const SparseMatrix& ) = delete;
         */

        using Storage::exists;

        scalar& at( size_type, size_type );

        using Storage::operator[];

        using Storage::begin;
        using Storage::end;

        size_type rows() const;
        size_type cols() const;
        size_type nonzeros() const;

        Storage& unsafe() { return *this; }
    };

    template<>
    template< typename B >
    SparseMatrix< Dune::BCRSMatrix< B > >
    ::SparseMatrix( const Storage& x ) : Storage( x ) {}

    template<>
    template< typename B >
    typename SparseMatrix< Dune::BCRSMatrix< B > >::scalar&
    SparseMatrix< Dune::BCRSMatrix< B > >
    ::at( size_type x, size_type y ) {
        return this->entry( x, y );
    }

    template<>
    template< typename B >
    typename SparseMatrix< Dune::BCRSMatrix< B > >::size_type
    SparseMatrix< Dune::BCRSMatrix< B > >::rows() const {
        return this->N();
    }

    template<>
    template< typename B >
    typename SparseMatrix< Dune::BCRSMatrix< B > >::size_type
    SparseMatrix< Dune::BCRSMatrix< B > >::cols() const {
        return this->M();
    }

    template<>
    template< typename B >
    typename SparseMatrix< Dune::BCRSMatrix< B > >::size_type
    SparseMatrix< Dune::BCRSMatrix< B > >::nonzeros() const {
        return this->nonzeroes();
    }
}

#endif //OPM_SPARSEMATRIX_DUNE

#endif //HAVE_DUNE_ISTL
