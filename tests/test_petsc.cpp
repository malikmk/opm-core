#include <vector>
#include <iostream>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/core/linalg/LinearSolverPetsc.hpp>

int main(int argc, char** argv)
{
    Opm::parameter::ParameterGroup param(argc, argv, false);
    //Initialize Eigen SpMat;
//    A =[ 0, 3, 0, 0, 0,
//         22, 0, 0, 0, 17,
//         7, 5, 0, 1, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 14, 0, 8];
    int size = 6;
    int nnz = 10;
    int ia[] = {0,1,3,6,7,9,10};
    int ja[] = {1,0,4,0,1,3,3,2,4,5};
    double sa[] = {3,22,17,7,5,1,1,14,8,8};
    double rhs[]={3,39,13,1,22,8};
    double sol[size];
    Opm::LinearSolverFactory linsolver(param);
    linsolver.solve(size, nnz, &ia[0], &ja[0], &sa[0], &rhs[0], &sol[0]);
    std::cout << "Solution is: \n";
    for (int i = 0; i < 6; ++i)
        std::cout << sol[i]<< " ";
    std::cout <<std::endl;
    return 0;
}
