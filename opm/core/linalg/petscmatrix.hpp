#ifndef OPM_PETSCMATRIX_H
#define OPM_PETSCMATRIX_H

#include <opm/core/linalg/petsc.hpp>

#include <petscmat.h>

#include <memory>
#include <vector>

namespace Opm {
namespace petsc {

    class vector;

    /// Class implementing C++-support for PETSc's Mat object. Provides no
    /// extra indirection and can be used as a handle for vanilla PETSc
    /// functions, should the need be.

    class matrix {
        public:
            typedef PetscScalar scalar;
            typedef PetscInt size_type;

            enum class nonzero_pattern { different, subset, same };

            /* Forward declaration of the (nested) builder class that is meant
             * to build in place of the constructors
             */
            class builder;

            /// Takes ownership of a Mat produced by some other means. The
            /// destruction and lifetime is now managed by matrix. Provided for
            /// compability with raw PETSc
            explicit matrix( Mat );
            /// Copy constructor
            matrix( const matrix& );

            matrix( const builder& );
            matrix( builder&& );

            /// Construct a (dense) matrix from std::vector
            /// Interprets the vector as a logical rows*columns 2D array
            /// \param[in] vector       Logically 2D std::vector to copy from
            /// \param[in] rows         Number of rows
            /// \param[in] columns      Number of columns
            matrix( const std::vector< scalar >&, size_type, size_type );

            /// Implicit conversion to Mat for compability with PETSc functions
            operator Mat() const;

            /// Get number of rows.
            /// \param[out] size    Number of rows in the matrix
            size_type rows() const;
            /// Get number of cols.
            /// \param[out] size    Number of columns in the matrix 
            size_type cols() const;

            /// Add a value to all elements in the matrix
            /// \param[in] scalar   Value to add to all elements
            //matrix& operator+=( scalar );
            /// Subtract a value from all elements in the matrix
            /// \param[in] scalar   Value to subtract from all elements
            //matrix& operator-=( scalar );
            /// Scalar multiplication
            /// \param[in] scalar   Value to scale with
            matrix& operator*=( scalar );
            /// Inverse scalar multiplication (division)
            /// \param[in] scalar   Value to scale with
            matrix& operator/=( scalar );

            /// Matrix addition, A + B.
            /// Equivalent to axpy( x, 1, different ). If you know your
            /// matrices have identical nonzero patterns, consider using axpy
            /// instead
            /// \param[in] matrix   Matrix to add
            matrix& operator+=( const matrix& );
            /// Matrix subtraction, A - B.
            /// Equivalent to axpy( x, -1, different ). If you know your
            /// matrices have identical nonzero patterns, consider using axpy
            /// instead
            /// \param[in] matrix   Matrix to add
            matrix& operator-=( const matrix& );

            /// A *= B, where A and B are matrices
            /// Equivalent to multiply( B );
            /// \param[in] B        The matrix B
            matrix& operator*=( const matrix& );

            /* these currently do not have to be friends (since matrix is
             * implicitly convertible to Mat), but they are for least possible
             * surprise
             */
            friend matrix operator*( matrix, scalar );
            friend matrix operator*( scalar, matrix );
            friend matrix operator/( matrix, scalar );

            friend matrix operator+( matrix, const matrix& );
            friend matrix operator-( matrix, const matrix& );
            /// C = A * B
            /// Equivalent to multiply( A, B )
            /// \param[in] A    The matrix A
            /// \param[in] B    The matrix B
            friend matrix operator*( const matrix&, const matrix& );

            /// y = Ax, matrix-vector multiplication
            /// Equivalent to multiply( A, x )
            friend vector operator*( const matrix&, const vector& );
            friend vector operator*( const vector&, const matrix& );

            /// Tests if two matrices are identical. This also checks matrix structure,
            /// so two identical sparse matrices with different nonzero structure but
            /// with explicit zeros will evaluate to false
            friend bool identical( const matrix&, const matrix& );

            /// A += aB
            /// \param[in] B        The matrix B
            /// \param[in] alpha    The constant alpha
            /// \param[in] pattern  The nonzero pattern relationship
            matrix& axpy( const matrix&, scalar, nonzero_pattern );

            /// A += B, equivalent to axpy( matrix, 1, pattern )
            /// \param[in] matrix   The matrix X
            /// \param[in] alpha    The constant alpha
            /// \param[in] pattern  The nonzero pattern relationship
            matrix& xpy( const matrix&, nonzero_pattern );

            /// A *= B, where A and B are matrices
            /// Assumes, if sparse, that A and B have the same nonzero pattern
            /// \param[in] B        The matrix B
            matrix& multiply( const matrix& );

            /// A *= B, where A and B are matrices
            /// Assumes, if sparse, that A and B have the same nonzero pattern
            /// \param[in] B        The matrix B
            /// \param[in] fill     The nonzero fill ratio
            /// nnz(C)/( nnz(A) + nnz(B) )
            matrix& multiply( const matrix&, scalar fill );

            /// In-place transposes the matrix, A <- A^T
            matrix& transpose();

            /// In-place hermitian transposes the matrix, A <- A^H
            matrix& hermitian_transpose();

        private:
            matrix();
            matrix( size_type, size_type );

            /* We rely on unique_ptr to manage lifetime semantics. */
            struct del { void operator()( Mat m ) { MatDestroy( &m ); } };
            std::unique_ptr< _p_Mat, del > m;

            template< typename T > T operator+( T );

            /* a convenient, explicit way to get the underlying Mat */
            inline Mat ptr() const;
    };

    /// Matrix-matrix multiplication. This is similar to operator*, but with
    /// more flexibility as you can tune the fill. The expected fill ratio of
    /// the multiplication C = A * B is nonzero(C)/(nonzero(A)+nonzero(B)).
    /// Defaults to letting PETSc decide.
    /// If experimenting, using PETSc's -info option can print the correct
    /// ratio (under Fill ratio)
    /// \param[in] A    Matrix A (left-hand side)
    /// \param[in] B    Matrix B (right-hand side)
    /// \param[in] fill (Optional) Matrix fill ratio
    matrix multiply( const matrix&, const matrix&, matrix::scalar = PETSC_DECIDE );

    /// Matrix-vector multiplication, y = Ax
    /// \param[in] A    The matrix
    /// \param[in] x    The vector
    vector multiply( const matrix&, const vector& );

    matrix transpose( const matrix& );
    matrix hermitian_transpose( matrix );

    class matrix::builder {
        public:
            typedef matrix::scalar scalar;
            typedef matrix::size_type size_type;

            builder() = delete;

            /// Constructor. PETSc -needs- to know the dimensions of the matrix
            /// beforehand, so a default constructor is not provided.
            builder( size_type, size_type );

            /// Copy constructor. Copies the currently built state. Particulary
            /// useful when several matrices share some structure, but diverges
            /// at a certain point
            builder( const builder& );

            /// Insert a single value. Defaults to 0 if you're only setting
            /// nonzero structure
            /// \param[in] row      The row to insert at
            /// \param[in] col      The col to insert at
            /// \param[in] value    (Optional) the value to insert.
            builder& insert( size_type, size_type, scalar = scalar() );

            /// Insert a full (sub)matrix in CSR format. This is more efficient
            /// than inserting single values. If the (sub)matrix sets some
            /// value previously set, that value will be overwritten by the
            /// latest.
            /// This might in the future be superseded by a stricter CSR type
            /// \param[in]      nonzero     The CSR nonzero vector
            builder& insert(    const std::vector< scalar >& nonzeros,
                                const std::vector< size_type >& row_indices,
                                const std::vector< size_type >& col_indices );

            /// Insert a row into the matrix. Inserts from [begin, begin + vec.size())
            /// This does not overwrite anything beyond vec.size()
            /// \param[in] row      Row to insert
            /// \param[in] values   Values to insert
            /// \param[in] begin    Where in the row to start insertion
            builder& insert_row(
                    size_type,
                    const std::vector< scalar >&,
                    size_type = 0 );

            /// Insert a row into the matrix at specific indices.
            /// This does not overwrite anything not touched by the column
            /// indices vector. Values must be equal in size to columns.
            /// \param[in] row      Row to insert
            /// \param[in] columns  Vector of column indices to insert
            /// \param[in] values   Values to insert
            builder& insert_row(    size_type,
                                    const std::vector< size_type >&,
                                    const std::vector< scalar >& );

            /// Commit the currently built structure into a completed matrix.
            /// This copies the currently built structure and returns a
            /// completed, ready-to-use matrix. Ideal for when several matrices
            /// have identical submatrices.
            matrix commit() const;

            /// An explicit move constructor. The builder object is left in a
            /// valid, but undefined state, i.e. you cannot keep adding values
            /// to it.
            matrix move();

        private:
            matrix m;

            builder& assemble();

            inline Mat ptr() const;

            friend class matrix;
    };

}
}

#endif //OPM_PETSCMATRIX_H
