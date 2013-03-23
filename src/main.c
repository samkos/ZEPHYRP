#include <petscdmda.h>
#include <petscksp.h>

#include <domain.h>
#include <init.h>
#include <poisson.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv){
  PetscErrorCode ierr;
  int nx = 63, ny = 63;
  DM dm;
  PetscBool flg;
  Mat A;
  Vec u, b;
  KSP solver;
  PC pc;
  double norm;
  PetscInt stage;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetInt(PETSC_NULL, "-nx", &nx, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-ny", &ny, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-assemble", &flg);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("preparing",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);

 
  ierr = PetscLogStageRegister("Domain creation",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = createDomain(&dm, nx, ny);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStageRegister("matrix creation",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = createMat(dm, &A, flg);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Vector creation",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
 
  ierr = DMCreateGlobalVector(dm, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(b, &u);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Domain initialisation",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = init2d(dm, b);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStageRegister("solver creation",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &solver);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(solver, "poisson_");CHKERRQ(ierr);
  ierr = KSPSetOperators(solver, A, A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(solver, KSPCG);
  ierr = KSPGetPC(solver, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(solver);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr);
  
  ierr = PetscLogStageRegister("Solving",&stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);

  ierr = KSPSolve(solver, b, u);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  //ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  //sleep(10);


  VecDestroy(&u);
  VecDestroy(&b);
  MatDestroy(&A);
  DMDestroy(&dm);
  KSPDestroy(&solver);
  ierr = PetscFinalize();
  return 0;
}
