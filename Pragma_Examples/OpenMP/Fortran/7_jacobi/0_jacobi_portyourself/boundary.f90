module boundary_mod
  use, intrinsic :: ISO_Fortran_env, only: int32, real64
  use mesh_mod, only: mesh_t
  implicit none

  private
  public :: boundary_conditions

contains

  subroutine boundary_conditions(mesh,u,au)
    type(mesh_t), intent(inout) :: mesh
    real(real64), intent(inout) :: u(:,:)
    real(real64), intent(inout) :: au(:,:)
    integer(int32) :: id,i,j,n_x,n_y
    real(real64) :: invdx2,invdy2

    n_x = mesh%n_x
    n_y = mesh%n_y
    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    do id=1,2*n_x+2*n_y-4
      if (id == 1) then
        au(1,1) = (2._real64*u(1,1)-u(2,1))*invdx2 &
                + (2._real64*u(1,1)-u(1,2))*invdy2
      else if (id <= n_x-1) then
        i = id
        au(i,1) = (-u(i-1,1)+2._real64*u(i,1)-u(i+1,1))*invdx2 &
                + (2._real64*u(i,1)-u(i,2))*invdy2
      else if (id == n_x) then
        au(n_x,1) = (2._real64*u(n_x,1)-u(n_x-1,1))*invdx2 &
                  + (2._real64*u(n_x,1)-u(n_x,2))*invdy2
      else if (id == n_x+1) then
        au(1,n_y) = (2._real64*u(1,n_y)-u(2,1))*invdx2 &
                  + (2._real64*u(1,n_y)-u(1,n_y-1))*invdy2
      else if (id <= 2*n_x-1) then
        i = id - n_x
        au(i,n_y) = (-u(i-1,n_y)+2._real64*u(i,n_y)-u(i+1,n_y))*invdx2 &
                  + (2._real64*u(i,n_y)-u(i,n_y-1))*invdy2
      else if (id == 2*n_x) then
        au(n_x,n_y) = (2._real64*u(n_x,n_y)-u(n_x-1,1))*invdx2 &
                    + (2._real64*u(n_x,n_y)-u(n_x,n_y-1))*invdy2
      else if (id <= 2*n_x+n_y-2) then
        j = id - 2*n_x + 1
        au(1,j) = (2._real64*u(1,j)-u(2,j))*invdx2 &
                + (-u(1,j-1)+2._real64*u(1,j)-u(1,j+1))*invdy2
      else
        j = id - 2*n_x - n_y + 3
        au(n_x,j) = (2._real64*u(n_x,j)-u(n_x-1,j))*invdx2 &
                  + (-u(n_x,j-1)+2._real64*u(n_x,j)-u(n_x,j+1))*invdy2
      end if
    end do

  end subroutine boundary_conditions

end module boundary_mod
