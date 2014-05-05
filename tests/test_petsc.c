#include <petsc.h>

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, (char*)0, "Testing Petsc Program!\n");
    PetscFinalize();
    
    printf("Hello Petsc.\n");
    return 0;
}
