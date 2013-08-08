//This program for solving single_phase flow
//
//\begin{eqnarray}
//\frac{1}{\partial t}\frac{\phi}{B_O}-\\
//\nabla\cdot(\frac{K}{\mu_o}\nabla p) = q_o in \Omega
//\frac{}\partial p}{\partial n} = 0 in \partial\Omega
//\end{eqnarray}
//
//$ID: miliu 07.08.2013 10:44:15 CST exp$
//$Email: miliu@statoil.com$
#include "config.h"

#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/io/vtk/writeVtkData.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <opm/core/props/IncompPropertiesBasic.hpp>
//#include <opm/core/linalg/LinearSolverUmfpack.hpp>
#include <opm/core/pressure/IncompTpfa.hpp>
#include <opm/core/pressure/FlowBCManager.hpp>
#include <opm/core/utility/miscUtilities.hpp>
#include <opm/core/utility/Units.hpp>

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>

#include <opm/core/simulator/TwophaseState.hpp>
#include <opm/core/simulator/WellState.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>

int 
main(int argc, char *argv[])
{
	using namespace Opm;
	std::cout<< "\n=================  Test program for single phase flow =================\n\n";
	parameter::ParameterGroup param(argc, argv, false);
	std::cout <<"-------------------  Reading parameters -------------------" << std::endl;
	//If we have a "deck_filename", grid and props will be read from that.
	bool use_deck = param.has("deck_filename");
	boost::scoped_ptr<EclipseGridParser> deck;
	boost::scoped_ptr<GridManager> grid;
	boost::scoped_ptr<IncompPropertiesInterface> props;
	boost::scoped_ptr<RockCompressibility> rock_comp;
	boost::scoped_ptr<Opm::WellsManager> wells;

	if (use_deck) {
		std::string deck_filename = param.get<std::string>("deck_filename");
		deck.reset(new EclipseGridParser(deck_filename));
		//Grid init
		grid.reset(new GridManager(*deck));
		//Rock and fluid init
		props.reset(new BlackoilPropertiesFromDeck(*deck, *grid->c_grid(), param));
 		rock_comp.reset(new RockCompressibility(*deck));
		gravity[2] = deck->hasField("NOGRAV") ? 0.0 : unit::gravity;

		//initial state variables (for single phase you may set so=1.0, sw=0)
		if (param.has("init_saturation")) {
			initStateBasic(*grid->c_grid(), *props, param, gravity[2], state);
		} else {
			initStateFromDeck(*grid->c_grid(), *props, *deck, gravity[2], state);
		}
	}	
}
