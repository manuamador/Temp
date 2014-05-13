const c = 299792458.
const mu0 = 4*pi*1e-7
const eps0 = 1/(mu0*c^2)


function p_rand (N, pmax=1.)
    """
    returns array of N random dipole moments (px,py,pz)
    |p| <= pmax
    """
    r=pmax*rand(N)
    phi=2*pi*rand(N)
    th=acos(2*rand(N).-1)
    xyz=([r.*sin(th).*cos(phi) r.*sin(th).*sin(phi) r.*cos(th)])
    return xyz
end

function R_rand (N, a=1.)
    """
    returns array of N random vectors (Rx,Ry,Rz)
    |R| = a
    """
    r=a*ones(N)
    phi=2*pi*rand(N)
    th=acos(2*rand(N).-1)
    xyz=([r.*sin(th).*cos(phi) r.*sin(th).*sin(phi) r.*cos(th)])
    return xyz
end



function Hertz_dipole (r, p, R, phi, f, t=0, epsr=1.)
    #=
    Calculate E field and B field strength of hertzian dipole(s)
    p: array of dipole moments
    R: array of dipole positions
    r: observation point
    f: frequency
    t: time
    phi: array with dipole phase angles (0..2pi)
    return: field strength at observation point r at time t (3-tuple: Ex, Ey, Ez)
    =#
    rprime=r-R  # r'=r-R
    magrprime=sqrt(sum(rprime.^2)) # |r-R|
    w=2*pi*f  # omega
    k=w/c     # wave number
    krp=k*magrprime  # k*|r-R|
    rprime_cross_p = cross(rprime, p) # (r-R) x p
    rp_c_p_c_rp = cross(rprime_cross_p, rprime) # ((r-R) x p) x (r-R)
    expfac=exp(1im*(-w*t+krp-phi))/(4*pi*eps0*epsr)
    E=expfac*(w^2/(c^2*magrprime^3)*rp_c_p_c_rp+(1/magrprime^3-w*im/(c*magrprime^2))*(3*rprime*dot(rprime,p)/magrprime^2-p))
	B=expfac/(magrprime*c^3)*(w^2*rprime_cross_p)/magrprime*(1-c/(im*w*magrprime))
    return E,B
  end


function Hertz_dipole_ff (r, p, R, phi, f, t=0, epsr=1.)
  #=
  Calculate E field and B field strength in far field of hertzian dipole(s)
  p: array of dipole moments
  R: array of dipole positions
  r: observation point
  f: frequency
  t: time
  phi: array with dipole phase angles (0..2pi)
  return: field strength at observation point r at time t (3-tuple: Ex, Ey, Ez)
  =#
  rprime=r-R  # r'=r-R
  magrprime=sqrt(sum(rprime.^2)) # |r-R|
  w=2*pi*f  # omega
  k=w/c     # wave number
  krp=k*magrprime  # k*|r-R|
  rprime_cross_p = cross(rprime, p) # (r-R) x p
  rp_c_p_c_rp = cross(rprime_cross_p, rprime) # ((r-R) x p) x (r-R)
  expfac=exp(1im*(-w*t+krp-phi))/(4*pi*eps0*epsr)
  E=expfac*(w^2/(c^2*magrprime^3)*rp_c_p_c_rp)
  B=expfac/(magrprime^2*c^3)*(w^2*rprime_cross_p)
  return E,B
end


function Hertz_dipole_nf (r, p, R, phi, f, t=0, epsr=1.)
    #=
  Calculate E field and B field strength in near field of hertzian dipole(s)
  p: array of dipole moments
  R: array of dipole positions
  r: observation point
  f: frequency
  t: time
  phi: array with dipole phase angles (0..2pi)
  return: field strength at observation point r at time t (3-tuple: Ex, Ey, Ez)
  =#
    rprime=r-R  # r'=r-R
    magrprime=sqrt(sum(rprime.^2)) # |r-R|
    w=2*pi*f  # omega
    k=w/c     # wave number
    krp=k*magrprime  # k*|r-R|
	rprime_cross_p = cross(rprime, p) # (r-R) x p
    expfac=exp(1im*(-w*t+krp-phi))/(4*pi*eps0*epsr)
    E=expfac*((1/magrprime^3-w*im/(c*magrprime^2))*(3*rprime*dot(rprime,p)/magrprime^2-p))
	B=expfac/(magrprime*c^2)*(w^2*rprime_cross_p)/magrprime*(-1/(im*w*magrprime))
    return E,B
	end

#main program
#EUT
freq= 2e9;#Frequency in Hz
a_EUT = 1     # radius of EUT
N_dipole = 100  # number of random dipoles
const Pow=1
const norm_p=sqrt(12*pi*c*Pow/(mu0*(2*pi*freq)^4))
ps=p_rand(N_dipole,norm_p)
Rs=R_rand(N_dipole, a_EUT)   # generate random dipole positions on EUT surface
phis=2*pi*rand(N_dipole) # generate random phases
ka=2*pi*freq/c*a_EUT; #electric size, ka=2*pi/lambda*a

#Radiation pattern
N=150;  #number of points along phi;
twoDsquareoverlambda=2*(a_EUT*2)^2*c/freq
nR=100
reduced_dist=logspace(-1,2,nR)
dist = twoDsquareoverlambda * reduced_dist.+a_EUT; #Measurement sphere radius in m
R=R_rand(N, 1) #measurement position

P=Array(Float64,N,nR); # Power matrix
for i=1:N
	println("$i")
	for j=1:nR
		Es=complex([0,0,0])
		for k=1:N_dipole
			E,B=Hertz_dipole( dist[j]*([R[i,1],R[i,2],R[i,3]]), [ps[k,1],ps[k,2],ps[k,3]], [Rs[k,1],Rs[k,2],Rs[k,3]],phis[k], freq) # calculate sum E-fields at obsevation points
			Es+=E
		end
		P[i,j]=sum(abs(Es).^2) # |E|**2
	end
end

D=maximum(P,1)./mean(P,1)

using PyPlot
plot(reduced_dist,D')
grid()
xscale("log")
ylim(0,maximum(D)*1.05)
