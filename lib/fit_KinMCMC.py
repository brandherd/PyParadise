# apply.c     "LIGHTHOUSE" NESTED SAMPLING APPLICATION
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# translated to Python by Issac Trotts in 2007
#
#              u=0                                 u=1
#               -------------------------------------
#          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
#              |::::::::::::::::::::::LIGHT::::::::::|
#         north|::::::::::::::::::::::HOUSE::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
# --*--------------*----*--------*-**--**--*-*-------------*--------
#             x=-2          coastline -->east      x=2
# Problem:
#  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.
# Inputs:
#  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
#  Prior(v)    is uniform (=1) over (0,1), mapped to y = 2*v; so that
#  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
#  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)
# Outputs:
#  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
#  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
#  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy

from math import *
import random
from mininest import nested_sampling

n=100                   # number of objects
max_iter = 1000         # number of iterations

class Object:
    def __init__(self):
        self.u=None     # Uniform-prior controlling parameter for x
        self.v=None     # Uniform-prior controlling parameter for y
        self.x=None     # Geographical easterly position of lighthouse
        self.y=None     # Geographical northerly position of lighthouse
        self.logL=None  # logLikelihood = ln Prob(data | position)
        self.logWt=None # log(Weight), adding to SUM(Wt) = Evidence Z

uniform = random.random;

def logLhood(   # logLikelihood function
    x,          # Easterly position
    y,spec,spec_model):         # Northerly position
      
    return logL

def sample_from_prior(limits):
    Obj = Object()
    random.seed()
    Obj.u = random.random()                # uniform in (0,1)
    Obj.v = random.random()                # uniform in (0,1)
    Obj.x = limits[0]+ Obj.u*(limits[1]-limits[0]) # map to x
    Obj.y = limits[2]+ Obj.v*(limits[3]-limits[0])                    # map to y
    Obj.logL = logLhood(Obj.x, Obj.y)
    return Obj

# Note that unlike the C version, this function returns an 
# updated version of Obj rather than changing the original.
def explore(   # Evolve object within likelihood constraint
    Obj,       # Object being evolved
    logLstar): # Likelihood constraint L > Lstar

    random.seed()

    ret = Object()
    ret.__dict__ = Obj.__dict__.copy()
    step = 0.1;   # Initial guess suitable step-size in (0,1)
    accept = 0;   # # MCMC acceptances
    reject = 0;   # # MCMC rejections
    Try = Object();          # Trial object

    for m in range(20):  # pre-judged number of steps

        # Trial object
        Try.u = ret.u + step * (2.*uniform() - 1.)  # |move| < step
        Try.v = ret.v + step * (2.*uniform() - 1.)  # |move| < step
        Try.u -= floor(Try.u)      # wraparound to stay within (0,1)
        Try.v -= floor(Try.v)      # wraparound to stay within (0,1)
        Try.x = limits[0]+ Try.u*(limits[1]-limits[0]  # map to x
        Try.y = limits[2]+ Try.v*(limits[3]-limits[0])        # map to y
        Try.logL = logLhood(Try.x, Try.y)  # trial likelihood value

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret.__dict__ = Try.__dict__.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):   step *= exp(1.0 / accept);
        if( accept < reject ):   step /= exp(1.0 / reject);
    return ret

def process_results(results):
    (x,xx) = (0.0, 0.0) # 1st and 2nd moments of x
    (y,yy) = (0.0, 0.0) # 1st and 2nd moments of y
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    for i in range(ni):
        w = exp(samples[i].logWt - logZ); # Proportional weight
        x  += w * samples[i].x;
        xx += w * samples[i].x * samples[i].x;
        y  += w * samples[i].y;
        yy += w * samples[i].y * samples[i].y;
    logZ_sdev = results['logZ_sdev']
    H = results['info_nats']
    H_sdev = results['info_sdev']
    print("# iterates: %i"%ni)
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
    print("Information: H  = %g nats = %g bits"%(H,H/log(2.0)))
    print("mean(x) = %9.4f, stddev(x) = %9.4f"%(x, sqrt(xx-x*x)));
    print("mean(y) = %9.4f, stddev(y) = %9.4f"%(y, sqrt(yy-y*y)));

if __name__ == "__main__":
    results = nested_sampling(n, max_iter, sample_from_prior, explore)
    process_results(results)


