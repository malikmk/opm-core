/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
  Copyright 2014 STATOIL ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_LINEARSOLVERPETSC_HEADER_INCLUDED
#define OPM_LINEARSOLVERPETSC_HEADER_INCLUDED
#include <opm/core/linalg/LinearSolverInterface.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <string>

namespace Opm
{


    /// Concrete class encapsulating some dune-istl linear solvers.
    class LinearSolverPetsc : public LinearSolverInterface
    {
    public:
        /// Default constructor.
        /// All parameters controlling the solver are defaulted:
        ///   linsolver_residual_tolerance  1e-8
        ///   linsolver_verbosity           0
        ///   linsolver_type                1 ( = CG_AMG), alternatives are:
        ///                                 CG_ILU0 = 0, CG_AMG = 1, BiCGStab_ILU0 = 2
        ///                                 FastAMG=3, KAMG=4 };
        ///   linsolver_save_system         false
        ///   linsolver_save_filename       <empty string>
        ///   linsolver_max_iterations      0 (unlimited=5000)
        ///   linsolver_residual_tolerance  1e-8
        ///   linsolver_smooth_steps        2
        ///   linsolver_prolongate_factor   1.6
        ///   linsolver_verbosity           0
        LinearSolverPetsc();
        LinearSolverPetsc(int argc, char** argv);

        /// Construct from parameters
        /// Accepted parameters are, with defaults, listed in the
        /// default constructor.
        LinearSolverPetsc(const parameter::ParameterGroup& param);

        /// Destructor.
        virtual ~LinearSolverPetsc();

        using LinearSolverInterface::solve;

        /// Solve a linear system, with a matrix given in compressed sparse row format.
        /// \param[in] size        # of rows in matrix
        /// \param[in] nonzeros    # of nonzeros elements in matrix
        /// \param[in] ia          array of length (size + 1) containing start and end indices for each row
        /// \param[in] ja          array of length nonzeros containing column numbers for the nonzero elements
        /// \param[in] sa          array of length nonzeros containing the values of the nonzero elements
        /// \param[in] rhs         array of length size containing the right hand side
        /// \param[inout] solution array of length size to which the solution will be written, may also be used
        ///                        as initial guess by iterative solvers.
        virtual LinearSolverReport solve(const int size,
                                         const int nonzeros,
                                         const int* ia,
                                         const int* ja,
                                         const double* sa,
                                         const double* rhs,
                                         double* solution) const;

        /// Set tolerance for the residual in dune istl linear solver.
        /// \param[in] tol         tolerance value
        virtual void setTolerance(const double tol);

        /// Get tolerance ofthe linear solver.
        /// \param[out] tolerance value
        virtual double getTolerance() const;
    private:
        int             argc_;
        char**          argv_;
        //enum KspType {richardson=0, chebyshev=1, cg=2, bicg=3, gmres=4, fgmres=5, dgmres=6, gcr=7, bcgs=8, cgs=9, tfqmr=10, tcqmr=11, cr=12, lsqr=13, preonly=14};
      //  KspType         ksp_type_;  /*ksp method*/
    //    enum PcType {jacobi=0, bjacobi=1, sor=2, eisenstat=3, icc=4, ilu=5, pcasm=6, gamg=7, ksp=8, composit=9, lu=10, cholesky=11, none=12};
        //PcType          pc_type_;
        //const std::string ksp_[];//={"richardson", "chebyshev", "cg", "bicg", "gmres", "fgmres", "dgmres", "gcr", "bcgs", "cgs", "tfqmr", "tcqmr", "cr", "lsqr", "preonly"};
      //  const std::string pc_[];// = {"jacobi", "bjacobi", "sor", "eisenstat", "icc", "ilu", "pcasm", "gamg", "ksp", "composit", "lu", "cholesky", "none"};
        std::string     ksp_type_;
        std::string     pc_type_;
        int             view_ksp_;
        double          rtol_;
        double          atol_;
        double          dtol_;
        int             maxits_;
    };


} // namespace Opm



#endif // OPM_LINEARSOLVERPETSC_HEADER_INCLUDED
