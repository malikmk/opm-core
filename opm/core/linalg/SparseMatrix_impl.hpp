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
    {
        assert( values );
        assert( col_ind );
        assert( row_ind );
    }

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

    template< typename Scalar >
    CSR< Scalar >::CSR(
            std::size_t rows, std::size_t nonzero,
            Scalar* values, int* col_ind, int* row_ind ) :
        unmanaged( rows, nonzero, values, col_ind, row_ind )
    {}

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

    template< typename Scalar, typename Storage >
    SparseMatrix< Scalar, Storage >&
    SparseMatrix< Scalar, Storage >::operator=(
        SparseMatrix< Scalar, Storage >&& mv ) {

        this->rows_ = mv.rows_;
        this->nonzeros_ = mv.nonzeros_;
        this->values_ = mv.values_;
        this->col_ind_ = mv.col_ind_;
        this->row_ind_ = mv.row_ind_;

        mv.release();
        return *this;
    }

    template< typename Scalar, typename Storage >
    SparseMatrix< Scalar, Storage >::SparseMatrix(
            std::size_t rows, std::size_t nonzero,
            Scalar* values, int* col_ind, int* row_ind ) :
        Storage( rows, nonzero, values, col_ind, row_ind )
    {
        assert( values );
        assert( col_ind );
        assert( row_ind );
    }

    template< typename Scalar, typename Storage >
    bool SparseMatrix< Scalar, Storage >::exists( std::size_t x, std::size_t y ) const {
        assert( x < this->rows() - 1 );

        static const unsigned int threshold = 0;

        const auto row_begin = this->col_ind_ + this->row_ind_[ x ];
        const auto row_end = this->col_ind_ + this->row_ind_[ x + 1 ];


        /*
         * Search the range [begin,end) and see if the y column is set
         */
        if( row_end - row_begin < threshold )
            return std::find( row_begin, row_end, y ) != row_end;

        return std::binary_search( row_begin, row_end, y );
    }

    template< typename Scalar, typename Storage >
    typename Storage::template row_ref< Scalar >
    SparseMatrix< Scalar, Storage >::operator[]( std::size_t x ) {
        assert( x < this->rows() - 1 );

        return *iterator( this->values_, this->col_ind_, this->row_ind_, x );
    }

    template< typename Scalar, typename Storage >
    typename Storage::template row_ref< const Scalar >
    SparseMatrix< Scalar, Storage >::operator[]( std::size_t x ) const {
        return this->operator[]( x );
    }

    template< typename Scalar, typename Storage >
    typename SparseMatrix< Scalar, Storage >::iterator
    SparseMatrix< Scalar, Storage >::begin() {
        return iterator( this->values_, this->col_ind_, this->row_ind_, 0 );
    }

    template< typename Scalar, typename Storage >
    typename SparseMatrix< Scalar, Storage >::iterator
    SparseMatrix< Scalar, Storage >::end() {
        return iterator( this->values_, this->col_ind_, this->row_ind_, this->rows() );
    }

    template< typename Scalar, typename Storage >
    typename SparseMatrix< Scalar, Storage >::const_iterator
    SparseMatrix< Scalar, Storage >::cbegin() const {
        return const_iterator( this->values_, this->col_ind_, this->row_ind_, 0 );
    }

    template< typename Scalar, typename Storage >
    typename SparseMatrix< Scalar, Storage >::const_iterator
    SparseMatrix< Scalar, Storage >::cend() const {
        return const_iterator( this->values_, this->col_ind_, this->row_ind_, this->rows() );
    }

    static inline std::size_t guess_cols( const int* col_ind, std::size_t len ) {
        /* We cannot really be sure how many columns the matrix should have
         * because that's information not really stored (in all cases,
         * anyways). However, in CSR & friends we can assume that there is an
         * element in the "last" column, reducing the problem to finding the
         * max value in the col_ind array. Information beyond this point is
         * never stored or touched anyway, so it shouldn't matter.
         *
         * This value as a boundary, however, can NOT be relied upon
         */
        return *std::max_element( col_ind, col_ind + len ) + 1;
    }

    template< typename Scalar, typename Storage >
    std::size_t SparseMatrix< Scalar, Storage >::rows() const {
         /* Assumes CSR-compatible storage. Use template specialization to override. */
        return this->rows_;
    }

    template< typename Scalar, typename Storage >
    std::size_t SparseMatrix< Scalar, Storage >::cols() const {
        return guess_cols( this->col_ind_, this->nonzeros_ );
    }

    template< typename Scalar, typename Storage >
    std::size_t SparseMatrix< Scalar, Storage >::nonzeros() const {
         /* Assumes CSR-compatible storage. Use template specialization to override. */
        return this->nonzeros_;
    }

    template< typename Scalar, typename Storage >
    typename Storage::unmanaged SparseMatrix< Scalar, Storage >::get() {
        return Storage::get();
    }

    template< typename Scalar, typename Storage >
    typename Storage::unmanaged SparseMatrix< Scalar, Storage >::release() {
        return Storage::release();
    }

    template< typename Scalar >
    void SparseMatrixBuilder< Scalar >::add( std::size_t i, std::size_t j, Scalar val ) {
        this->sorted_ = false;
        this->values_.push_back( COO{ i, j, val } );
    }

    template< typename Scalar >
    SparseMatrix< Scalar > SparseMatrixBuilder< Scalar >::to_csr() {
        /* Sort the vector of (x, y, val) triples. This makes building the CSR
         * representation simple - the first pass copies the values and column
         * indices into the destination arrays. The second pass calculates the
         * row pointers (aka the offsets into the val/col arrays).
         *
         * This simple copy only works when the source list is sorted
         */
        if( !this->sorted_ )
            std::sort( this->values_.begin(), this->values_.end() );

        this->values_.erase( std::unique( this->values_.begin(), this->values_.end() ), this->values_.end() );

        /* sequential calls won't re-sort unless necessary */
        this->sorted_ = true;

        /* the CSR format only represents to the deepest row with a nonzero in
         * it
         *
         * TODO: consider support for explicitly representing more trailing
         * zero-rows
         *
         * we add 2 to the highest found row index in order to have room for
         * the end-pointer (so not to special-case the final delta) and to
         * account for off-by-one errors in allocation
         */
        const std::size_t rows = this->values_.back().row_ + 2;

        std::unique_ptr< Scalar[] > values( new Scalar[ this->values_.size() ] );
        std::unique_ptr< int[] > col_ind( new int[ this->values_.size() ] );
        std::unique_ptr< int[] > row_ind( new int[ rows ] );

        /* first pass: copy column indices and nonzero values */
        auto* curr_col = col_ind.get();
        auto* curr_val = values.get();
        for( const auto val : this->values_ ) {
            *curr_col++ = val.col_;
            *curr_val++ = val.value_;
        }

        /* second pass: build the row indices array.
         *
         * The range of rows are represented through the delta rows[ i ]
         * rows[ i + 1 ]. If a row is empty then rows[ i ] == rows[ i + 1 ]
         * (which both then points to the same offset in vals/cols)
         */
        auto* curr_row = row_ind.get();
        *curr_row = 0;
        for( unsigned int i = 0; i < this->values_.size(); ++i ) {
            assert( curr_row > row_ind.get() );

            /* determine what index we're currently at */
            const std::size_t row_pos = std::distance( row_ind.get(), curr_row );
            const auto& val = this->values_[ i ];

            /* no advancement, we're still on the same row */
            if( row_pos == val.row_ ) continue;

            /* fill all deltas between previously set row_ind and this with i -
             * this handles the case of an empty row
             */
            std::fill_n( curr_row + 1, val.row_ - row_pos, i );
            curr_row += val.row_ - row_pos;
        }

        row_ind[ rows - 1 ] = this->values_.size();

        return SparseMatrix< Scalar >( rows - 1, this->values_.size(),
                values.release(), col_ind.release(), row_ind.release() );
    }
}

#endif //OPM_SPARSEMATRIX_IMPL
