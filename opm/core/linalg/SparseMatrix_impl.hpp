#ifndef OPM_SPARSEMATRIX_IMPL
#define OPM_SPARSEMATRIX_IMPL

#include <algorithm>
#include <cassert>
#include <iostream>

namespace Opm {

    template< typename Scalar >
    unmanaged::CSR< Scalar >::CSR(
            std::size_t rows, std::size_t nonzeros,
            Scalar* values, int* col_ind, int* row_ind ) :
        rows_( rows ),
        nonzeros_( nonzeros ),
        values_( values ),
        col_ind_( col_ind ),
        row_ind_( row_ind )
    {}

    template< typename Scalar >
    template< typename T, typename Derived >
    Derived CSR< Scalar >::base_itr< T, Derived >::operator++( int ) {
        Derived itr( *this );
        this->operator++();
        return itr;
    }

    template< typename Scalar >
    template< typename T, typename Derived >
    Derived CSR< Scalar >::base_itr< T, Derived >::operator--( int ) {
        Derived itr( *this );
        this->operator--();
        return itr;
    }

    template< typename Scalar >
    template< typename T, typename Derived>
    bool CSR< Scalar >::base_itr< T, Derived >::operator!=(
        const Derived& other ) const {
        return !static_cast< const Derived* >( this )->operator==( other );
    }

    template< typename Scalar >
    template< typename T, typename Derived>
    T* CSR< Scalar >::base_itr< T, Derived >::operator->() {
        return &this->operator*();
    }

    template< typename Scalar >
    template< typename T >
    CSR< Scalar >::col_itr< T >::col_itr(
            Scalar* values,
            int* col_ind,
            std::size_t index ) :
        values_( values ),
        col_ind_( col_ind ),
        index_( index )
    {}

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template col_itr< T >&
    CSR< Scalar >::col_itr< T >::operator++() {
        ++this->index_;
        return *this;
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template col_itr< T >&
    CSR< Scalar >::col_itr< T >::operator--() {
        --this->index_;
        return *this;
    }

    template< typename Scalar >
    template< typename T >
    bool CSR< Scalar >::col_itr< T >::operator==(
        const CSR< Scalar >::col_itr< T >& other ) const {
        return this->index_ == other.index_;
    }

    template< typename Scalar >
    template< typename T >
    T& CSR< Scalar >::col_itr< T >::operator*() {
        return this->values_[ this->index_ ];
    }

    template< typename Scalar >
    template< typename T >
    std::size_t CSR< Scalar >::col_itr< T >::index() const {
        return this->col_ind_[ this->index_ ];
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template col_itr< T >
    CSR< Scalar >::row_ref< T >::begin() {
        return col_itr< T >(
                this->values_,
                this->col_ind_,
                this->row_ind_[ this->index_ ] );
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template col_itr< T >
    CSR< Scalar >::row_ref< T >::end() {
        const auto offset = this->row_ind_[ this->index_ + 1 ];
        return col_itr< T >( this->values_, this->col_ind_, offset );
    }

    template< typename Scalar >
    template< typename T >
    T& CSR< Scalar >::row_ref< T >::operator[]( std::size_t x ) {
        /*
         * Assumes the matrix coordinate is nonzero - returns result anyways,
         * but behaviour is undefined if matrix coordinate has an (implicit)
         * zero.
         */
        const auto begin = this->col_ind_ + this->row_ind_[ this->index_ ];
        const auto end = this->col_ind_ + this->row_ind_[ this->index_ + 1 ];

        const auto target = std::lower_bound( begin, end, x );

        /* for debug builds we check the validity of the returned value
         * if *target != x then the searched-for matrix coordinate is implicit
         * zero, i.e. not set in the matrix. this is an error and we won't
         * guarantee correct results. if this check is necessary during
         * run-time, use at()
         */
        assert( *target == x );

        return this->values_[ std::distance( this->col_ind_, target ) ];
    }

    template< typename Scalar >
    template< typename T >
    const T& CSR< Scalar >::row_ref< T >::operator[]( std::size_t x ) const {
        return this->operator[]( x );
    }

    template< typename Scalar >
    template< typename T >
    std::size_t CSR< Scalar >::row_ref< T >::index() const {
        return this->index_;
    }

    template< typename Scalar >
    template< typename T >
    std::size_t CSR< Scalar >::row_ref< T >::size() const {
        return this->row_ind_[ this->index_ + 1 ] - this->row_ind_[ this->index_ ];
    }

    template< typename Scalar >
    template< typename T >
    CSR< Scalar >::row_ref< T >::row_ref(
            Scalar* values,
            int* col_ind,
            int* row_ind,
            std::size_t index ) :
        values_( values ),
        col_ind_( col_ind ),
        row_ind_( row_ind ),
        index_( index )
    {}

    template< typename Scalar >
    template< typename T >
    CSR< Scalar >::row_itr< T >::row_itr(
            Scalar* values,
            int* col_ind,
            int* row_ind,
            std::size_t index ) :
        row_ref< T >( values, col_ind, row_ind, index )
    {}

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template row_itr< T >&
    CSR< Scalar >::row_itr< T >::operator++() {
        ++this->index_;
        return *this;
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template row_itr< T >&
    CSR< Scalar >::row_itr< T >::operator--() {
        --this->index_;
        return *this;
    }

    template< typename Scalar >
    template< typename T >
    bool CSR< Scalar >::row_itr< T >::operator==(
        const CSR< Scalar >::row_itr< T >& other ) const {
        return this->index_ == other.index_;
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template row_ref< T >&
    CSR< Scalar >::row_itr< T >::operator*() {
        return *this;
    }

    template< typename Scalar >
    template< typename T >
    typename CSR< Scalar >::template row_ref< T >*
    CSR< Scalar >::row_itr< T >::operator->() {
        return this;
    }

    /*
     * Explicitly default-construct (i.e. 0) all fields. This ensures that no
     * value is ever uninitialized, even in the POD unmanaged::CSR struct
     */
    template< typename Scalar >
    CSR< Scalar >::CSR() : unmanaged( 0, 0, NULL, NULL, NULL ) {}

    template< typename Scalar >
    CSR< Scalar >::CSR(
            std::size_t rows, std::size_t nonzero,
            Scalar* values, int* col_ind, int* row_ind ) :
        unmanaged( rows, nonzero, values, col_ind, row_ind )
    {
        assert( values );
        assert( col_ind );
        assert( row_ind );
    }

    template< typename Scalar >
    CSR< Scalar >::CSR( CSR< Scalar >&& mv ) :
        unmanaged( mv.release() )
    {}

    template< typename Scalar >
    CSR< Scalar >::~CSR() {
        if( this->values_ ) delete[] this->values_;
        if( this->col_ind_ ) delete[] this->col_ind_;
        if( this->row_ind_ ) delete[] this->row_ind_;
    }

    template< typename Scalar >
    typename unmanaged::CSR< Scalar >
    CSR< Scalar >::get() {
        return *this;
    }

    template< typename Scalar >
    typename unmanaged::CSR< Scalar >
    CSR< Scalar >::release() {
        unmanaged x = *this;
        this->values_ = NULL;
        this->col_ind_ = this->row_ind_ = NULL;
        return x;
    }

    template< typename Storage >
    SparseMatrix< Storage >&
    SparseMatrix< Storage >::operator=(
        SparseMatrix< Storage >&& mv ) {

        this->rows_ = mv.rows_;
        this->nonzeros_ = mv.nonzeros_;
        this->values_ = mv.values_;
        this->col_ind_ = mv.col_ind_;
        this->row_ind_ = mv.row_ind_;

        mv.release();
        return *this;
    }

    template< typename Storage >
    Storage& SparseMatrix< Storage >::unsafe() {
        return *this;
    }

    template< typename Scalar, typename Storage >
    void SparseMatrixBuilder< Scalar, Storage >::add( std::size_t i, std::size_t j, Scalar val ) {
        this->insert( std::make_pair( CO{ i, j }, val ) );
    }

    template< typename Scalar, typename Storage >
    template< typename Return >
    SparseMatrix< Return > SparseMatrixBuilder< Scalar, Storage >::convert() {
        return conv< Scalar, Storage, Return >::convert( this );
    }
}

#endif //OPM_SPARSEMATRIX_IMPL
