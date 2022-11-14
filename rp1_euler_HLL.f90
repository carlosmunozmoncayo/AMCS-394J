! =========================================================
subroutine rp1(maxmx,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =========================================================

! solve Riemann problems for the 1D Euler equations using the HLLE
! approximate Riemann solver.

! waves: 2
! equations: 3

! Conserved quantities:
!       1 V
!       2 u
!       3 epsilon

    implicit none

    integer, intent(in) :: maxmx, meqn, mwaves, mbc, mx, maux
    double precision, dimension(meqn,1-mbc:maxmx+mbc), intent(in) :: ql, qr
    double precision, dimension(maux,1-mbc:maxmx+mbc), intent(in) :: auxl, auxr
    double precision, dimension(meqn, mwaves, 1-mbc:maxmx+mbc), intent(out) :: wave
    double precision, dimension(meqn, 1-mbc:maxmx+mbc), intent(out) :: amdq, apdq
    double precision, dimension(mwaves, 1-mbc:maxmx+mbc), intent(out) :: s

    double precision :: ul, ur, um
    double precision :: pl, pr, pm
    double precision :: Vl, Vr, Vm
    double precision :: epsl, epsr, epsm
    double precision :: dl, dr
    double precision :: gamma, gamma1
    double precision :: s1, s2
    double precision, dimension(3) :: q_l, q_r, q_m !Local vectors of conserved quantities
    double precision, dimension(3) :: fql, fqr
    integer :: m, i, mw


    common /cparam/  gamma

    gamma1 = gamma - 1.d0

    do i=2-mbc,mx+mbc
        !local vectors of conserved quantities
        q_l = qr(:,i-1)
        q_r = ql(:,i  )
        ! New variables 
        Vl = q_l(1) 
        Vr = q_r(1)
        ! Velocity
        ul = q_l(2)
        ur = q_r(2)
        ! New variable (epsilon) 
        epsl = q_l(3)
        epsr = q_r(3)  
        !Pressure
        pl = gamma1*(epsl-0.5d0*ul**2)/Vl
        pr = gamma1*(epsr-0.5d0*ur**2)/Vr
        !Flux at left and right states
        fql = (/ -ul, pl, ul*pl /)
        fqr = (/ -ur, pr, ur*pr /)
        !Smaller and larger eigenvales evaluated at left and right states
        dl = dsqrt(pl/Vl)
        dr = dsqrt(pr/Vr)
        !Defining some simple HLL speeds (neglecting contact discontinuity)
        s1 = -dl
        s2 = dr
        !Middle state HLL
        q_m = (1.d0/(s1-s2))*(fqr-fql-s2*q_r+s1*q_l)
        !Defining waves for Clawpack
        wave(:,1,i) = q_m-q_l
        wave(:,1,i) = q_r-q_m
        !Defining speeds for Clawpack
        s(1,i) = s1
        s(2,i) = s2
       !wave(1,1,i) = rho_m - qr(1,i-1)
       !wave(2,1,i) = rhou_m - qr(2,i-1)
       !wave(3,1,i) = E_m - qr(3,i-1)
       !s(1,i) = s1
    
       !wave(1,2,i) = ql(1,i) - rho_m
       !wave(2,2,i) = ql(2,i) - rhou_m
       !wave(3,2,i) = ql(3,i) - E_m
       !s(2,i) = s2

    end do 

    do m=1,3
        do i=2-mbc, mx+mbc
            amdq(m,i) = 0.d0
            apdq(m,i) = 0.d0
            do mw=1,mwaves
                if (s(mw,i) < 0.d0) then
                    amdq(m,i) = amdq(m,i) + s(mw,i)*wave(m,mw,i)
                else
                    apdq(m,i) = apdq(m,i) + s(mw,i)*wave(m,mw,i)
                endif
            end do
        end do
    end do

end subroutine rp1
