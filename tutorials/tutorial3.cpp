/*
  Copyright 2012 SINTEF ICT, Applied Mathematics.

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


#if HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cassert>
#include <opm/core/grid.h>
#include <opm/core/GridManager.hpp>
#include <opm/core/utility/writeVtkData.hpp>
#include <opm/core/linalg/LinearSolverUmfpack.hpp>
#include <opm/core/pressure/IncompTpfa.hpp>
#include <opm/core/pressure/FlowBCManager.hpp>
#include <opm/core/fluid/IncompPropertiesBasic.hpp>

#include <opm/core/transport/reorder/TransportModelTwophase.hpp>

#include <opm/core/simulator/TwophaseState.hpp>

#include <opm/core/utility/miscUtilities.hpp>
#include <opm/core/utility/Units.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

/// \page tutorial3 Multiphase flow
/// The Darcy law gives 
///   \f[u_\alpha= -\frac1{\mu_\alpha} K_\alpha\nabla p_\alpha\f]
/// where \f$\mu_\alpha\f$ and \f$K_\alpha\f$ represent the viscosity
/// and the permeability tensor for each phase \f$\alpha\f$. In the two phase
/// case, we have either \f$\alpha=w\f$ or \f$\alpha=o\f$. 
/// In this tutorial, we do not take into account capillary pressure so that 
/// \f$p=p_w=p_o\f$ and gravity
/// effects. We  denote by \f$K\f$ the absolute permeability tensor and each phase
/// permeability is defined through its relative permeability by the expression
/// \f[K_\alpha=k_{r\alpha}K.\f]
/// The phase mobility are defined as
///  \f[\lambda_\alpha=\frac{k_{r\alpha}}{\mu_\alpha}\f]
/// so that the Darcy law may be rewritten as
///  \f[u_\alpha= -\lambda_\alpha K\nabla p.\f]
/// The conservation of mass for each phase writes:
/// \f[\frac{\partial}{\partial t}(\phi\rho_\alpha s_\alpha)+\nabla\cdot(\rho_\alpha u_\alpha)=q_\alpha\f]
/// where \f$s_\alpha\f$ denotes the saturation of the phase \f$\alpha\f$ and \f$q_\alpha\f$ is a source term. Let
/// us consider a two phase flow with oil and water. We assume that the rock and both fluid phases are incompressible. Since
/// \f$s_w+s_o=1\f$, we may add the conservation equations to get
///  \f[ \nabla\cdot u=\frac{q_w}{\rho_w}+\frac{q_o}{\rho_o}.\f]
/// where we define
///   \f[u=u_w+u_o.\f]
/// Let the total mobility be equal to
/// \f[\lambda=\lambda_w+\lambda_o\f]
/// Then, we have 
/// \f[u=-\lambda K\nabla p.\f]
/// The set of equations
/// \f[\nabla\cdot u=\frac{q_w}{\rho_w}+\frac{q_o}{\rho_o},\quad u=-\lambda K\nabla p.\f]
/// is referred to as the <strong>pressure equation</strong>. We introduce 
/// the fractional flow \f$f_w\f$
/// as
/// \f[f_w=\frac{\lambda_w}{\lambda_w+\lambda_o}\f]
/// and obtain
/// \f[\phi\frac{\partial s_w}{\partial t}+\nabla\cdot(f_w u)=\frac{q_w}{\rho_w}\f]
/// which is referred to as the <strong>transport equation</strong>. The pressure and 
/// transport equation are coupled. In this tutorial, we implement a splitting scheme, 
/// where, at each time step, we decouple the two equations. We solve first
/// the pressure equation and then update the water saturation by solving
/// the transport equation assuming that \f$u\f$ is constant in time in the time step 
/// interval we are considering.



/// \page tutorial3
/// \details
/// Main function
/// \code
int main ()
{
    /// \endcode
    /// \page tutorial3
    /// \details
    /// We define the grid. A cartesian grid with 400 cells,
    /// each being 10m along each side. Note that we treat the
    /// grid as 3-dimensional, but have a thickness of only one
    /// layer in the Z direction.
    ///
    /// The Opm::GridManager is responsible for creating and destroying the grid,
    /// the UnstructuredGrid data structure contains the actual grid topology
    /// and geometry.
    /// \code
    int nx = 20;
    int ny = 20;
    int nz = 1;
    double dx = 10.0;
    double dy = 10.0;
    double dz = 10.0;
    using namespace Opm;
    GridManager grid_manager(nx, ny, nz, dx, dy, dz);
    const UnstructuredGrid& grid = *grid_manager.c_grid();
    int num_cells = grid.number_of_cells;
    /// \endcode

    /// \page tutorial3
    /// \details
    /// We define the properties of the fluid.\n 
    /// Number of phases, phase densities, phase viscosities,
    /// rock porosity and permeability.
    ///
    /// We always use SI units in the simulator. Many units are
    /// available for use, however.  They are stored as constants in
    /// the Opm::unit namespace, while prefixes are in the Opm::prefix
    /// namespace. See Units.hpp for more.
    /// \code
    int num_phases = 2;
    using namespace Opm::unit;
    using namespace Opm::prefix;
    std::vector<double> density(num_phases, 1000.0);
    std::vector<double> viscosity(num_phases, 1.0*centi*Poise);
    double porosity = 0.5;
    double permeability = 10.0*milli*darcy;
    /// \endcode

    /// \page tutorial3
    /// \details We define the relative permeability function. We use a basic fluid
    /// description and set this function to be linear. For more realistic fluid, the 
    /// saturation function may be interpolated from experimental data.
    /// \code
    SaturationPropsBasic::RelPermFunc rel_perm_func = SaturationPropsBasic::Linear;
    /// \endcode

    /// \page tutorial3
    /// \details We construct a basic fluid and rock property object
    /// with the properties we have defined above.  Each property is
    /// constant and hold for all cells.
    /// \code
    IncompPropertiesBasic props(num_phases, rel_perm_func, density, viscosity,
                                porosity, permeability, grid.dimensions, num_cells);
    /// \endcode

    /// \page tutorial3
    /// \details Gravity parameters. Here, we set zero gravity.
    /// \code
    const double *grav = 0;
    std::vector<double> omega; 
    /// \endcode

    /// \page tutorial3
    /// \details We may now set up the pressure solver. At this point,
    /// unchanging parameters such as transmissibility are computed
    /// and stored internally by the IncompTpfa class. The final (null pointer)
    /// constructor argument is for wells, which are now used in this tutorial.
    /// \code
    LinearSolverUmfpack linsolver;
    IncompTpfa psolver(grid, props.permeability(), grav, linsolver, 0);
    /// \endcode

    /// \page tutorial3
    /// \details We set up the source term. Positive numbers indicate that the cell is a source,
    /// while negative numbers indicate a sink.
    /// \code
    std::vector<double> src(num_cells, 0.0);
    src[0] = 1.;
    src[num_cells-1] = -1.;
    /// \endcode

    /// \page tutorial3
    /// \details We set up data vectors for the wells. Here, there are
    /// no wells and we let them be empty dummies.
    /// \code
    std::vector<double> empty_wdp;
    std::vector<double> empty_well_bhp;
    std::vector<double> empty_well_flux;
    /// \endcode


    /// \page tutorial3
    /// \details We compute the pore volume
    /// \code
    std::vector<double> porevol;
    Opm::computePorevolume(grid, props.porosity(), porevol);
    /// \endcode

    /// \page tutorial3
    /// \details Set up the transport solver. This is a reordering implicit Euler transport solver.
    /// \code
    const double tolerance = 1e-9;
    const int max_iterations = 30;
    Opm::TransportModelTwophase transport_solver(grid, props, tolerance, max_iterations);
    /// \endcode

    /// \page tutorial3
    /// \details Time integration parameters
    /// \code
    const double dt = 0.1*day;
    const int num_time_steps = 20;
    /// \endcode


    /// \page tutorial3
    /// \details We define a vector which contains all cell indexes. We use this 
    /// vector to set up parameters on the whole domain.
    /// \code
    std::vector<int> allcells(num_cells);
    for (int cell = 0; cell < num_cells; ++cell) {
        allcells[cell] = cell;
    }
    /// \endcode

    /// \page tutorial3
    /// \details We set up the boundary conditions. Letting bcs be empty is equivalent 
    /// to no-flow boundary conditions.
    /// \code
    FlowBCManager bcs;
    /// \endcode


    /// \page tutorial3
    /// \details 
    /// We set up a two-phase state object, and
    /// initialise water saturation to minimum everywhere.
    /// \code
    TwophaseState state;
    state.init(grid, 2);
    state.setFirstSat(allcells, props, TwophaseState::MinSat);
    /// \endcode
    
    /// \page tutorial3
    /// \details We introduce a vector which contains the total mobility 
    /// on all cells.
    /// \code
    std::vector<double> totmob;
    /// \endcode

    /// \page tutorial3
    /// \details This string stream will be used to construct a new
    /// output filename at each timestep.
    /// \code
    std::ostringstream vtkfilename;
    /// \endcode


    /// \page tutorial3
    /// \details Loop over the time steps.
    /// \code
    for (int i = 0; i < num_time_steps; ++i) {
        /// \endcode
        /// \page tutorial3

        /// \details Compute the total mobility. It is needed by the
        /// pressure solver and must be recomputed every time step
        /// since it depends on the saturation.
        /// \code
        computeTotalMobility(props, allcells, state.saturation(), totmob);
        /// \endcode
        /// \page tutorial3
        /// \details Solve the pressure equation
        /// \code
        psolver.solve(totmob, omega, src, empty_wdp, bcs.c_bcs(), 
                      state.pressure(), state.faceflux(), empty_well_bhp, 
                      empty_well_flux);
        /// \endcode
        /// \page tutorial3
        /// \details  Solve the transport equation.
        /// \code
        transport_solver.solve(&state.faceflux()[0], &porevol[0], &src[0],
                               dt, state.saturation());
        /// \endcode

        /// \page tutorial3
        /// \details Write the output to file.
        /// \code
        vtkfilename.str(""); 
        vtkfilename << "tutorial3-" << std::setw(3) << std::setfill('0') << i << ".vtu";
        std::ofstream vtkfile(vtkfilename.str().c_str());
        Opm::DataMap dm;
        dm["saturation"] = &state.saturation();
        dm["pressure"] = &state.pressure();
        Opm::writeVtkData(grid, dm, vtkfile);
    }
}
/// \endcode



/// \page tutorial3
/// \section results3 Results.
/// <TABLE>
/// <TR>
/// <TD> \image html tutorial3-000.png </TD>
/// <TD> \image html tutorial3-005.png </TD>
/// </TR>
/// <TR>
/// <TD> \image html tutorial3-010.png </TD>
/// <TD> \image html tutorial3-015.png </TD>
/// </TR>
/// <TR>
/// <TD> \image html tutorial3-019.png </TD>
/// <TD> </TD>
/// </TR>
/// </TABLE>


/// \page tutorial3
/// \details
/// \section completecode3 Complete source code:
/// \include tutorial3.cpp 
/// \include generate_doc_figures.py 
