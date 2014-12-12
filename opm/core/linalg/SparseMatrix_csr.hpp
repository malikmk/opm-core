#ifndef OPM_SPARSEMATRIX_CSR
#define OPM_SPARSEMATRIX_CSR

/*
 * Opm::CSR<> specializations for SparseMatrix.
 */

namespace Opm {

    namespace {

        /* Put in an anonymous namespace because it can be considered a private
         * function.
         *
         * This snippet is a bit syntax heavy.
         *
         * SparseMatrixBuilder has a convert() method that must be specialized
         * for the relevant combinations of underlying storage and target type.
         * Since functions in C++ cannot be partially specialized, this has to
         * be accomplished via a "worker" class - struct conv.
         *
         * We want partially specialized functions because of the fundamental
         * type parameter. While somewhat hard to understand, they enable a
         * neat final interface for the -user-.
         *
         * The first template<> line is us telling the compiler that this is a
         * fully specialized instance of the conv class.
         *
         * template< Scalar > means that this specialization is parametrised on
         * Scalar.
         *
         * Then the actually struct *re-implementation* for this specific type.
         * This is a C++ specification detail, and for why this is moved to its
         * own, independent class.
         *
         * The SparseMatrixBuilder has struct conv as a friend, meaning the
         * obj* passed is equivalent to this*.
         *
         * None of this is visible in user code. The external interface is
         * builder.convert< target_type >();
         */

    template<>
    template< typename Scalar >
    struct conv< Scalar, std::map< CO, Scalar >, CSR< Scalar > > {
    static SparseMatrix< CSR< Scalar > >
        convert( SparseMatrixBuilder< Scalar, std::map< CO, Scalar > >* obj ) {

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
        const std::size_t rows = obj->rbegin()->first.row_ + 2;

        std::unique_ptr< Scalar[] > values( new Scalar[ obj->size() ] );
        std::unique_ptr< int[] > col_ind( new int[ obj->size() ] );
        std::unique_ptr< int[] > row_ind( new int[ rows ] );

        /* first pass: copy column indices and nonzero values */
        auto* curr_col = col_ind.get();
        auto* curr_val = values.get();
        for( const auto val : *obj ) {
            *curr_col++ = val.first.col_;
            *curr_val++ = val.second;
        }

        /* second pass: build the row indices array.
         *
         * The range of rows are represented through the delta rows[ i ]
         * rows[ i + 1 ]. If a row is empty then rows[ i ] == rows[ i + 1 ]
         * (which both then points to the same offset in vals/cols)
         */
        auto* curr_row = row_ind.get();
        *curr_row = 0;
        auto valitr = obj->cbegin();
        for( unsigned int i = 0; i < obj->size(); ++i, ++valitr ) {
            assert( curr_row > row_ind.get() );

            /* determine what index we're currently at */
            const std::size_t row_pos = std::distance( row_ind.get(), curr_row );
            const auto& val_row = valitr->first.row_;

            /* no advancement, we're still on the same row */
            if( row_pos == val_row ) continue;

            /* fill all deltas between previously set row_ind and obj with i -
             * obj handles the case of an empty row
             */
            std::fill_n( curr_row + 1, val_row - row_pos, i );
            curr_row += val_row - row_pos;
        }

        row_ind[ rows - 1 ] = obj->size();

        return SparseMatrix< CSR< Scalar > >( rows - 1, obj->size(),
                values.release(), col_ind.release(), row_ind.release() );
    }

    };

    }

    /*
     * CSR-specific re-implementation of the SparseMatrix class. This is picked
     * whenver SparseMatrix< CSR< double > > is requested.
     */
    template<>
    template< typename Scalar >
    class SparseMatrix< CSR< Scalar > >
    : private CSR< Scalar > {
        private:
            typedef CSR< Scalar > Storage;

        public:
            /* typedefs can be forwarded directly from CSR */
            typedef Scalar type;
            typedef Scalar scalar;

            typedef typename Storage::size_type size_type;

            typedef typename Storage::unmanaged unmanaged;

            typedef typename Storage::iterator iterator;
            typedef typename Storage::const_iterator const_iterator;

            typedef typename Storage::template row_ref< Scalar > row_ref;
            typedef typename Storage::template row_ref< const Scalar > const_row_ref;

        SparseMatrix() = default;
        SparseMatrix( size_type rows, size_type nonzero,
                Scalar* values, int* col_ind, int* row_ind );

        /*
         * A test to see if the (x,y) coordinate is nonzero. log(n) complexity
         * for CSR
         */
        bool exists( size_type, size_type ) const;

        /* bounds checked operator[][] (Not implemented yet) */
        Scalar& at( size_type, size_type );

        row_ref operator[]( size_type );
        const_row_ref operator[]( size_type ) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        size_type rows() const;
        /*
         * cols() does -not- actually guarantee returning the correct result
         * for CSR storage. The reason for this is that there could be columns
         * all filled with zeros which aren't represented (and really not very
         * useful) that are supposed to be there, but aren't. For most cases
         * this should be irrelevant, and this estimate is ok.
         *
         * O(n) with the size of nonzeros.
         */
        size_type cols() const;
        size_type nonzeros() const;

        Storage& unsafe() { return *this; }
    };

    template<>
    template< typename Scalar >
    SparseMatrix< CSR< Scalar > >::SparseMatrix(
            size_type rows, size_type nonzero,
            Scalar* values, int* col_ind, int* row_ind ) :
        CSR< Scalar >( rows, nonzero, values, col_ind, row_ind )
    {
        assert( values );
        assert( col_ind );
        assert( row_ind );
    }


    template<>
    template< typename Scalar >
    bool SparseMatrix< CSR< Scalar > >
    ::exists( size_type x, size_type y ) const {
        assert( x < this->rows() - 1 );

        /* Not determined yet - set threshold for when to do linear search
         * instead of binary search
         */
        const unsigned int threshold = 0;

        const auto row_begin = this->col_ind_ + this->row_ind_[ x ];
        const auto row_end = this->col_ind_ + this->row_ind_[ x + 1 ];


        /*
         * Search the range [begin,end) and see if the y column is set
         */
        if( row_end - row_begin < threshold )
            return std::find( row_begin, row_end, y ) != row_end;

        return std::binary_search( row_begin, row_end, y );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::row_ref
    SparseMatrix< CSR< Scalar > >::operator[]( size_type x ) {
        assert( x < this->rows() - 1 );

        return *iterator( this->values_, this->col_ind_, this->row_ind_, x );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::const_row_ref
    SparseMatrix< CSR< Scalar > >::operator[]( size_type x ) const {
        return this->operator[]( x );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::iterator
    SparseMatrix< CSR< Scalar > >::begin() {
        return iterator( this->values_, this->col_ind_, this->row_ind_, 0 );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::iterator
    SparseMatrix< CSR< Scalar > >::end() {
        return iterator( this->values_, this->col_ind_, this->row_ind_, this->rows() );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::const_iterator
    SparseMatrix< CSR< Scalar > >::cbegin() const {
        return const_iterator( this->values_, this->col_ind_, this->row_ind_, 0 );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::const_iterator
    SparseMatrix< CSR< Scalar > >::cend() const {
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

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::size_type
    SparseMatrix< CSR< Scalar > >::rows() const {
        return this->rows_;
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::size_type
    SparseMatrix< CSR< Scalar > >::cols() const {
        return guess_cols( this->col_ind_, this->nonzeros_ );
    }

    template<>
    template< typename Scalar >
    typename SparseMatrix< CSR< Scalar > >::size_type
    SparseMatrix< CSR< Scalar > >::nonzeros() const {
        return this->nonzeros_;
    }
}

#endif //OPM_SPARSEMATRIX_CSR
