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
    double precision :: x_switch_RS
    double precision :: p0, V0, fpl, fpr, a0


    common /cparam/  gamma, x_switch_RS 

    gamma1 = gamma - 1.d0



    do i=2-mbc,mx+mbc
        !local vectors of conserved quantities
        q_l = qr(:,i-1)
        q_r = ql(:,i  )
        ! Specific volume 1 over density
        Vl = q_l(1) 
        Vr = q_r(1)
        ! Velocity
        ul = q_l(2)
        ur = q_r(2)
        ! Energy over density (epsilon) 
        epsl = q_l(3)
        epsr = q_r(3)  
        !Pressure
        pl = gamma1*(epsl-0.5d0*ul**2)/Vl
        pr = gamma1*(epsr-0.5d0*ur**2)/Vr

        if (auxr(1,i-1) <= x_switch_RS) then
            !!!!!!!!!!!!!!!
            !Solve the full Euler equations in Lagrangian coordinates
            !!!!!!!!!!!!!!

            !Flux at left and right states
            fql(1) = -ul
            fql(2) = pl
            fql(3) = ul*pl
            fqr(1) = -ur
            fqr(2) = pr
            fqr(3) = ur*pr
            !Smaller and larger eigenvales evaluated at left and right states
            dl = dsqrt(pl/Vl)
            dr = dsqrt(pr/Vr)
            !Defining some simple HLL speeds (neglecting contact discontinuity)
            s1 = -max(dl,dr)
            s2 = max(dl,dr)
            !Middle state HLL
            q_m = (1.d0/(s1-s2))*(fqr-fql-s2*q_r+s1*q_l)
        else
            !!!!!!!!!!!!!!!!
            !Solve Burger's equations
            !Since we require constant density and pressure we take the ones from the left state
            !!!!!!!!!!!!!!!!
            p0 = pl
            V0 = Vl
            a0 = dsqrt(gamma*p0*V0)
            !Derivative of Burger's flux at left and right states
            fpl = dsqrt(gamma*p0/V0)*(1.d0+0.5d0*gamma1*ul/a0)**((gamma+1.d0)/gamma1)
            fpr = dsqrt(gamma*p0/V0)*(1.d0+0.5d0*gamma1*ur/a0)**((gamma+1.d0)/gamma1)
            !For this problem we have a single wave, so we just need one speed  
            if (abs(fpl)>abs(fpr)) then
                s2 = fpl
            else
                s2 = fpr
            end if
            !In this case the left wave will propagate no information
            s1 = 0.d0
            q_m = q_l 
        end if

            wave(:,1,i) = q_m-q_l
            wave(:,2,i) = q_r-q_m
            !Defining speeds for Clawpack
            s(1,i) = s1
            s(2,i) = s2
       
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
