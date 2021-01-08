from __future__ import division
import math, random
import numpy as np
from scipy.optimize import fsolve
from numpy import genfromtxt
from scipy import interpolate
import matplotlib.pyplot as plt
import pylab

random.seed(0)

DopplerSpectrum = []

# Fundamental Constants
c = 299792458 # Meters per Second; Speed of Light
h = 6.62607*math.pow(10, -34) # Joules * Seconds; Planck's constant
bMagneton = 9.27401*math.pow(10, -24) # Ampere * Meters^2; Bohr Magneton
kboltz = 1.38065*math.pow(10,-23) # Joules per Kelvin; Boltzmann Constant

# Dy Constants
m = 2.6983757*math.pow(10,-25) # Kilograms; Mass of Single Dysprosium Atom
T = 1523.2 # Convert 1250 C to Kelvin; Temperature of Oven

# Zeeman Slower Beam Constants  
wavelength = 421*math.pow(10, -9) # Meters; Transition Wavelength
k = 2*math.pi/(wavelength) # Meters^-1; Wavenumber for Transition 
gamma = 2*math.pi*30*math.pow(10, 6) # Hertz; Excited State Linewidth
vp = (10)*h*k/(math.pi*2*m) # Meters per Second; Recoil velocity from a single photon
v0 = 483.5 # Meters per Second; Velocity Group We Cool From; 483.5 is Most Probable
B0 = 82*math.pow(10,-3) # Ideally for Zero Velocity: h*v0/(wavelength*bMagneton) Tesla; Magnetic Field at Entrance to Zeeman Slower; 77 mT for 30 m/s final velocity
timestep = (10)*2/gamma # Seconds; The Natural Time Step Unit in our Monte Carlo Simulation
#B_Bias = 1*math.pow(10,-2) # Magnetic Field Bias in Zeeman Slower
sat = 3.5 # Dimensionless; Saturation Intensity Parameter, I/I_saturation, for ZS Beam
ZSLaserDetuning = -1.46*math.pow(10,9) #-.15*math.pow(10,9) #-1.46*math.pow(10,9) #-.16*math.pow(10,9) #-1.46*math.pow(10,9) (sat = 10) #-1.3272*math.pow(10,9) (sat = 1) #-1.1485*math.pow(10,9) #- B_Bias*(bMagneton)/h # Hertz; Frequency Detuning of Laser c/lambda from Resonance


# Lengths
#LZS = .42 # 42 cm; Length of Zeeman Slower Tube
LOPA = .1 # 10 cm; Length from Oven to Primary Aperature
LOTC = LOPA + .03 # 8 cm; Length from Oven to Transverse Cooling Beams
LODP = LOTC + 0.04 # 12 cm; Length from Oven to Differential Pump Entry
LOZS = LODP + 0.1 # 22 cm; Length from Oven to Zeeman Slower
LZSMOT = .67 # 67 cm; Length from Beginning of Zeeman Slower to MOT Center
LTotal = LOZS + LZSMOT # Meters; Length of entire Simulation from Oven to MOT Center

# Radii
OAR = .0015 # 1.5 mm; Oven Aperature Radius
PAR = .0035 # 3.5 mm; Primary Aperature Radius
DPA = .0035 # 3.5 mm; Differential Pump Aperature Radius
ZSA = .0125 # Zeeman Slower Aperture

# Transverse Cooling Beam Constants
TCW = 0 #.03 # 3 cm; Transverse Cooling Beam Width
TCLaserDetuning = -1*gamma/(2*math.pi) # Hertz; Transverse Cooling Laser Detuning
TCISat = np.pi*h*c*gamma/(3*math.pow(wavelength,3)) #Dimensionless; Transverse Cooling Saturation Intensity; Lifetime = 1/gamma
TCI = 320.5 # (100 mW)/(10.4 mm * 3 cm); Transverse Cooling Laser Intensity
TCSatParameter = TCI/TCISat # Dimensionless; Transverse Cooling Saturation Parameter

# Li's Magnetic Field Profile
BData = genfromtxt('C:\Users\Cantara\Documents\Physics\Dypole\ZeemanSlower\MeasuredBFieldProfile_September2018Simulation.csv', delimiter=',')

l = []
b = []
for x in range(0,751):
    l = np.append(l, BData[x][0])

for x in range(0,751):
    b = np.append(b, BData[x][1])

f = interpolate.interp1d(l, b)   
 
#def Bfield(z, L): # Determine magnetic field at coordinate z
#    return B0*(1-np.sqrt(1-z/L))

def detuning(B, vz, laserdetuning): # Determine total detuning from resonance for a given velocity
    return k*vz + 2*math.pi*(laserdetuning + bMagneton*abs(B)/h)

def F(v): # Units of v are meters per second
    return (v**3)*((m/(2*math.pi*kboltz*T))**2)*2*(math.pi**2)*math.exp(-m*(v**2)/(2*kboltz*T)) #Outputs units of Seconds per Meter (need to multiply by dv to get dimensionless probability)

def scatteringprob(detuned, saturation): # Probability for an atom to absorb the photon
    return saturation/(1+saturation+4*(detuned**2/(gamma**2)))
    
def emissionTheta():
    n = random.random()
    return np.arccos(math.pow(-2+4*n+math.sqrt(5-16*n+16*n**2), -1/3) - math.pow(-2+4*n+math.sqrt(5-16*n+16*n**2), 1/3))

def StartParameters(vinit, count):
    Go = False
    while Go == False:
        theta_init = random.random()*2*math.pi
        r_init = math.sqrt(random.random())*OAR
        theta_oven = ((PAR+OAR)/LOPA)*math.sqrt(random.random()) # np.arctan((PAR+OAR)/LOPA)*math.sqrt(random.random)
        phi_oven = 2*math.pi*random.random()
        x = r_init*np.sin(theta_init)
        y = r_init*np.cos(theta_init)
        vx = vinit*np.sin(theta_oven)*np.cos(phi_oven) 
        vy = vinit*np.sin(theta_oven)*np.sin(phi_oven)
        vz = vinit*np.cos(theta_oven)
        z = 0 # Starting at oven aperture
        
        while z < LOPA: # While atoms are traveling towards primary aperture for collimation
            B = 0
            ZSDetuned = detuning(B, vz, ZSLaserDetuning) # Determine overall detuning from both Doppler shift, Zeeman splitting and detuned laser light
            scatprob = scatteringprob(ZSDetuned,sat) # Determine probability for a photon-atom scatter from Zeeman Slowing beam 
            
            if vz <= 0: # If atoms turn around, count them as lost
                break 
                        
            if scatprob > random.random(): # If an absorption occurs then...
                theta = emissionTheta()
                phi = 2*np.pi*random.random()
                vx = 0#vx - vp*np.sin(theta)*np.cos(phi) # Corresponding velocity kicks from photon absorption and emission
                vy = 0#vy - vp*np.sin(theta)*np.sin(phi)
                vz = vz - vp*(1+np.cos(theta))
                    
            x = x + vx*timestep # Update velocities based on scattering events and step forward in time
            y = y + vy*timestep        
            z = z + vz*timestep  
            
            if z > LOPA and math.pow(x,2) + math.pow(y,2) > math.pow(PAR,2): # If the atom does not make it through the primary aperture discard it and try again
                break
                
            if z > LOPA and math.pow(x,2) + math.pow(y,2) < math.pow(PAR,2): # If the atom does make it through the primary aperture use it
                count = count + 1
                Go = True
                break
    #print math.sqrt(x**2 + y**2) - PAR, PAR            
    return (count, vx, vy, vz, x, y, z)
                    
def TCScatters(vx, vy, vz, x, y, z): 
    B = 0
    detunedx = detuning(B, vx, TCLaserDetuning) # Determine overall detuning from both Doppler shift and detuned laser light
    scatprobx = scatteringprob(detunedx, TCSatParameter) #  Determine probability for a photon-atom scatter 
    detunedy = detuning(B, vy, TCLaserDetuning) # Determine overall detuning from both Doppler shift and detuned laser light
    scatproby = scatteringprob(detunedy, TCSatParameter) #  Determine probability for a photon-atom scatter
    detunedxm = detuning(B, -vx, TCLaserDetuning) # Determine overall detuning from both Doppler shift and detuned laser light
    scatprobxm = scatteringprob(detunedxm, TCSatParameter) #  Determine probability for a photon-atom scatter 
    detunedym = detuning(B, -vy, TCLaserDetuning) # Determine overall detuning from both Doppler shift and detuned laser light
    scatprobym = scatteringprob(detunedym, TCSatParameter) #  Determine probability for a photon-atom scatter
    ZSDetuned = detuning(B, vz, ZSLaserDetuning) # Determine overall detuning from both Doppler shift, Zeeman splitting and detuned laser light
    scatprobz = scatteringprob(ZSDetuned,sat) #  Determine probability for a photon-atom scatter from Zeeman Slowing beam    
    
    Arr1 = [scatprobx, scatprobxm, scatproby, scatprobym, scatprobz]
    Arr2 = []
    
    for t in xrange(0,5):
        if Arr1[t] > random.random():
            Arr2.append(t)
            
    if len(Arr2) == 0:
        return vx, vy, vz
    else:
        select = Arr2[int(math.floor(random.random()*len(Arr2)))]
        theta = emissionTheta()
        phi = 2*np.pi*random.random()
    
        if select == 4: 
            vx = vx - vp*np.sin(theta)*np.cos(phi) # Corresponding velocity kicks from photon absorption and emission
            vy = vy - vp*np.sin(theta)*np.sin(phi)
            vz = vz - vp*(1+np.cos(theta))     
                        
        elif select == 3:
            vx = vx - vp*np.sin(theta)*np.cos(phi)
            vy = vy - vp*(-1+np.sin(theta)*np.sin(phi))
            vz = vz - vp*np.cos(theta)
            
        elif select == 2:
            vx = vx - vp*np.sin(theta)*np.cos(phi)
            vy = vy - vp*(1+np.sin(theta)*np.sin(phi))
            vz = vz - vp*np.cos(theta)
    
        elif select == 1:
            vx = vx - vp*(-1+np.sin(theta)*np.cos(phi))
            vy = vy - vp*np.sin(theta)*np.sin(phi)
            vz = vz - vp*np.cos(theta)
            
        elif select == 0:
            vx = vx - vp*(1+np.sin(theta)*np.cos(phi))
            vy = vy - vp*np.sin(theta)*np.sin(phi)
            vz = vz - vp*np.cos(theta)
        return vx, vy, vz
                            
def PAToZS(vx, vy, vz, x, y, z):            
    while z < LOZS:

        if vz <= 0: # If atoms turn around, count them as lost
            PassThrough = False
            return (PassThrough, vx, vy, vz, x, y, z)  
                
        if z > LOTC and z < LOTC + TCW:
            vx, vy, vz = TCScatters(vx, vy, vz, x, y, z)
            
        else:
            if z >= LOZS - .07: # Experimentally Measured B-field values begin here
                B = f(z-LOZS-.03)
            else: 
                B = 0
            ZSDetuned = detuning(B, vz, ZSLaserDetuning) # Determine overall detuning from both Doppler shift, Zeeman splitting and detuned laser light
            print z, vz, B, ZSDetuned/(2*math.pi*30000000)
            scatprobz = scatteringprob(ZSDetuned,sat) # Determine probability for a photon-atom scatter from Zeeman Slowing beam  
            if scatprobz > random.random():  
                theta = emissionTheta()
                phi = 2*np.pi*random.random()
                vx = vx - vp*np.sin(theta)*np.cos(phi) # Corresponding velocity kicks from photon absorption and emission
                vy = vy - vp*np.sin(theta)*np.sin(phi)
                vz = vz - vp*(1+np.cos(theta))    
            
        x = x + vx*timestep # Update velocities based on scattering events and step forward in time
        y = y + vy*timestep        
        z = z + vz*timestep  
            
        if z > LODP and math.pow(x,2) + math.pow(y,2) > math.pow(DPA,2): # If the atom travels outside of ZS beam after Trasverse Cooling Stage discard it
            PassThrough = False
            return (PassThrough, vx, vy, vz, x, y, z)
            
        elif z > LOZS - 2*vz*timestep: #and math.pow(x,2) + math.pow(y,2) < math.pow(DPA,2): # If the atom makes it to the ZS, continue onwards
            PassThrough = True
            z = LOZS
            return (PassThrough, vx, vy, vz, x, y, z)   
    
def ZS(vx, vy, vz, x, y, z):
    while z < LTotal: # While atoms are inside Zeeman slowing pipe and until they reach MOT Beams  
        B = f(z-LOZS-.03) #+ B_Bias # Determine the B-field at a given z coordinate
        if z > LTotal - .20:
            #B = -(z-LTotal-.12)*(.082/.12) + .082
            
           print z, vz, B, (k*vz + 2*math.pi*(ZSLaserDetuning + bMagneton*abs(B)/h))/(2*math.pi*30000000)
        detuned = detuning(B, vz, ZSLaserDetuning) # Determine overall detuning from both Doppler shift, Zeeman splitting and detuned laser light
        scatprob = scatteringprob(detuned, sat) # Determine probability for a photon-atom scatter
        if vz <= 0:
            print "stopped and turned around with left:" 
            print (LTotal-z)
            return False
                        
        if scatprob > random.random():
            theta = emissionTheta() 
            phi = 2*np.pi*random.random()
            vx = vx - vp*np.sin(theta)*np.cos(phi)
            vy = vy - vp*np.sin(theta)*np.sin(phi)
            vz = vz - vp*(1+np.cos(theta))
                
        x = x + vx*timestep
        y = y + vy*timestep        
        z = z + vz*timestep
                
        if vz < 30 and math.pow(x,2) + math.pow(y,2) < math.pow(ZSA,2) and z > LTotal - 2*vz*timestep: # If the atom does make it through the exit of the Zeeman Slower with the appropriate velocity
            print "Cooled"
            print vz
            DopplerSpectrum.append(vz)
            return True
                    
        if vz > 30 and z > LTotal - 2*vz*timestep: # If the atom does make it through the exit of the Zeeman Slower with the appropriate velocity
            print "Too fast"
            print vz
            DopplerSpectrum.append(vz)
            return False
                
        if math.pow(x,2) + math.pow(y,2) > math.pow(ZSA,2):
            return False

samples = 1 # Dimensionless; Number of Runs for each Velocity Group  
totalcooled = 0
for vinit in xrange(140, 250): # Go through range of initial velocities coming from oven
    cooled = 0
    count = 0
    while count < samples: # Go through n = samples trials of particles for the velocity class
        count, vx, vy, vz, x, y, z = StartParameters(vinit, count)
        PassThrough, vx, vy, vz, x, y, z = PAToZS(vx, vy, vz, x, y, z)  
        if PassThrough and ZS(vx, vy, vz, x, y, z):
            cooled = cooled + 1
        print vinit           
    fcooled = cooled/samples # Fraction cooled for this velocity group
    totalcooled = totalcooled + fcooled*F(vinit) # Total fraction cooled of entire velocity distribution 
print ZSLaserDetuning, totalcooled
plt.hist(DopplerSpectrum)
plt.show()     
        