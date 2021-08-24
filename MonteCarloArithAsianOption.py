from scipy.stats import norm
import numpy as np
import sys

class MonteCarloArithAsianOption:
    """
      Monte Carlo method with or without control variate technique for arithmetic Asian call/put options
        S = spot price
        K = strike price
        t = time 0
        T = time to maturity
        r = risk free interest rate
        sigma = volatility of underlying asset
        n = # of observation times
        m = # of paths in simulation
        optionType = call or put
        ctrlVar = with or without control variate
    """
    def __init__(self, s0=None, r=0, T=0, K=None, sigma=None, n=100, m=100000, optionType=None, ctrlVar=False):
      try:
          self.s0 = s0
          self.sigma = sigma
          self.r = r
          self.T = T
          self.K = K
          self.n = n
          self.m = m
          self.optionType = optionType
          self.ctrlVar = ctrlVar
          self.dt = self.T / self.n
      except ValueError:
          print('Error passing Options parameters')
      if optionType != 'call' and optionType != 'put':
          raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
      if s0 < 0 or r < 0 or T <= 0 or K < 0 or sigma < 0:
          raise ValueError('Error: Negative inputs not allowed')

    def MonteCarloPrice(self):
        n, m, dt, df= self.n, self.m, self.dt, np.exp(-self.r * self.T)
        sigsqT = self.sigma**2 * self.T * (n + 1) * (2 * n + 1) / (6 * n * n)
        muT = 0.5 * sigsqT + (self.r - 0.5 * (self.sigma**2)) * self.T * (n+1)/(2*n)
        drift = np.exp((self.r - 0.5 * self.sigma**2) * dt)
        d1 = (np.log(self.s0 / self.K) + (muT + 0.5 * sigsqT)) / np.sqrt(sigsqT)
        d2 = d1 - np.sqrt(sigsqT)
        N1, N2, N1_, N2_ = norm.cdf(d1), norm.cdf(d2), norm.cdf(-d1), norm.cdf(-d2)
        arithPayoffCall, arithPayoffPut, geoPayoffCall, geoPayoffPut = [0] * m, [0] * m, [0] * m, [0] * m                       
        '''
            Options can be priced by Monte Carlo simulation.
        '''
        ### Generate M paths of the asset prices at time t1, · · · , tn, repeated runs to find control parameter ###
        # compute an N(0,1) sample ξi
        for i in range(m):
            # initial values of pseudo numbers is a seed value to generates same sequence of random variables when pseudo number generators run so results reproducible
            initState = i * 6
            Z = np.random.seed(initState)
            # Simulating random variables from distribution initiated by first generating a random number from a uniform distribution (0,1)
            Z = np.random.normal(0, 1, n)
            ### si = s0e^(r-0.5σ²)dt + σ√Tξi continuous asset model ###
            SPricePath = [0] * n
            # but given expression above we can easily get samples of X by generate samples of Z plug into expression
            growthFactor = drift * np.exp(self.sigma * np.sqrt(dt) * Z[0])
            # creates list for each price path
            SPricePath[0] = self.s0 * growthFactor
            # X is a random variable that can be simulated and let g(X) be a function that can be evaluated at the realizations of X, simulation generates multiple copies of g(X)
            # compute an N(0,1) sample ξj of price paths
            '''
                First, the price of the underlying asset is simulated by random number generation for a number of paths.
            '''
            for j in range(1, n):
                # on each increment the price update uses an N(0, 1) random variable j coming from the i.i.d. sequence
                growthFactor = drift * np.exp(self.sigma * np.sqrt(dt) * Z[j])
                SPricePath[j] = SPricePath[j-1] * growthFactor
            '''
                Then, the value of the option is found by calculating the average of discounted returns over all paths.
            '''
            #### Arithmatic mean ###
            # Arithmetic mean calculate mean by sum of total of values divided by number of values
            arithMean = np.mean(SPricePath)
            arithPayoffCall[i] = df * max(arithMean - self.K, 0)
            arithPayoffPut[i] = df * max(self.K-arithMean, 0) 

            #### Geometric mean ###
            # Geometric mean calculate mean or average of series of values of product which takes into account the effect of compounding and it is used for determining the performance of investment
            geoMean = np.exp((1 / n) * sum(np.log(SPricePath)))
            geoPayoffCall[i] = df * max(geoMean - self.K, 0)
            geoPayoffPut[i] = df * max(self.K-geoMean, 0)
        
        callPriceMean, putPriceMean = np.mean(arithPayoffCall), np.mean(arithPayoffPut)
        #### without Control Variate ###
        if not self.ctrlVar:
          print('Mente Carlo method without control variate.')
          callPriceStd, putPriceStd = np.std(arithPayoffCall), np.std(arithPayoffPut)
          
          # if we generate many samples of Y , then 95% of the samples fall in the range [−1.96, 1.96]
          if self.optionType == 'call':
            # the 95% confidence interval for call option
            lowerBound, upperBound = callPriceMean-1.96 * callPriceStd / np.sqrt(m), callPriceMean + 1.96 * callPriceStd / np.sqrt(m)
            confidentInterval = (lowerBound, upperBound)
            print('The call option price is {} with {} confidence interval.'.format(str(round(callPriceMean, 8)), confidentInterval))
            return callPriceMean, confidentInterval
        
          elif self.optionType == 'put':
            # the 95% confidence interval for put option
            lowerBound, upperBound = putPriceMean - 1.96 * putPriceStd / np.sqrt(m), putPriceMean + 1.96 * putPriceStd / np.sqrt(m)
            confidentInterval = (lowerBound, upperBound)
            print('The put option price is {} with {} confidence interval.'.format(str(round(putPriceMean, 8)), confidentInterval))
            return putPriceMean, confidentInterval
        
        #### with Control Variate ###
        if self.ctrlVar:
            # variance covariance matrix
            print('Mente Carlo method with control variate.')
            geoCallPriceMean = np.mean(geoPayoffCall)
            geoPutPriceMean = np.mean(geoPayoffPut)
            # cov(X,Y) = E[XY]−(EX)(EY)
            convXYCall = np.mean(np.multiply(arithPayoffCall, geoPayoffCall)) - (callPriceMean * geoCallPriceMean)
            convXYPut = np.mean(np.multiply(arithPayoffPut, geoPayoffPut)) - (putPriceMean * geoPutPriceMean)
            # Θ = cov(X,Y) / var(Y)
            thetaCall = convXYCall / np.var(geoPayoffCall)
            thetaPut = convXYPut / np.var(geoPayoffPut)
            
            # e^-rt(S0e^ρT N(d1) − KN(d2))
            if self.optionType == 'call':
                geoCall = df * (self.s0 * np.exp(muT) * N1 - self.K * N2)
                Z = arithPayoffCall + thetaCall * (geoCall - geoPayoffCall)
            elif self.optionType == 'put':
                geoPut = df * (self.K * N2_ - self.s0 * np.exp(muT) * N1_)
                Z = arithPayoffPut + thetaPut * (geoPut - geoPayoffPut)

            Zmean, Zstd = np.mean(Z), np.std(Z)
            lowerBound, upperBound = Zmean-1.96 * Zstd / np.sqrt(m), Zmean+1.96*Zstd/np.sqrt(m)
            confidentInterval = (lowerBound, upperBound)
            print('The {} option price is {} with {} confidence interval.'.format(self.optionType, str(round(Zmean, 8)), confidentInterval))
            return Zmean, confidentInterval

if __name__ == '__main__':
    option = MonteCarloArithAsianOption(s0=100, r=0.05, T=3, K=100, sigma=0.3, n=100, m=100000, optionType='put', ctrlVar=False)
    option.MonteCarloPrice()
