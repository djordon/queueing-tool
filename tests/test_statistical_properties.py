import math
import os
import unittest

import numpy as np

import queueing_tool as qt


TRAVIS_TEST = os.environ.get('TRAVIS_TEST', False)

def empirical_cdf0(x, z, n):
    return np.sum(z <= x) / n

empirical_cdf = np.vectorize(empirical_cdf0, excluded={1, 2})


def chi2_cdf(q, k, n=1000000, ns=1):
    return np.mean([empirical_cdf(q, np.random.chisquare(k, n), n) for i in range(ns)])

reason = "Test takes long."

class TestQueueServers(unittest.TestCase):

    def setUp(self):
        self.lam = np.random.randint(1,10) + 0.0
        self.rho = np.random.uniform(0.5, 1)

    @unittest.skipIf(TRAVIS_TEST, reason)
    def test_Markovian_QueueServer(self):

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)

        arr = lambda t: t + np.random.exponential(1/self.lam)
        ser = lambda t: t + np.random.exponential(1/mu)

        q = qt.QueueServer(nServers=nSe, arrival_f=arr, service_f=ser)
        n = 50000

        q.set_active()
        q.simulate(n=20000)    # Burn in period
        q.collect_data = True
        q.simulate(nD=n+1)
        dat = q.fetch_data()
        dep = dat[:, 2][dat[:, 2] > 0]
        dep.sort()
        dep = dep[1:] - dep[:-1]
        q.clear()

        n = len(dep)
        lamh = 1 / np.mean(dep)
        upb  = - np.log( 6.0 / n) / lamh
        nbin = n // 6 - 1 #np.floor( np.exp( lam * upb) - 1 )
        bins = np.zeros(nbin+2)
        bins[1:-1] = upb - np.log( np.arange(nbin, 0, -1)) / lamh
        bins[-1]   = np.infty

        N  = np.histogram(dep, bins=bins)[0]
        pp = 1 - np.exp(-lamh * bins)
        pr = n*(pp[1:] - pp[:-1])
        Q  = np.sum((N - pr)**2 / pr)
        p1 = 1 - chi2_cdf(Q, nbin-1)

        x, y = dep[1:], dep[:-1]
        cc   = np.corrcoef(x,y)[0,1]
        self.assertAlmostEqual( cc, 0, 1)
        self.assertTrue( p1 > 0.05 )

    @unittest.skipIf(TRAVIS_TEST, reason)
    def test_QueueServer_Littleslaw(self):

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)

        arr = lambda t: t + np.random.exponential(1 / self.lam)
        ser = lambda t: t + np.random.exponential(1 / mu)

        q = qt.QueueServer(nServers=nSe, arrival_f=arr, service_f=ser)
        n = 500000

        q.set_active()
        q.simulate(n=n)    # Burn in period
        q.collect_data = True
        q.simulate(n=n+1)
        data = q.fetch_data()
        q.clear()

        ind  = data[:,2] > 0
        souj = data[ind, 2] - data[ind, 0]
        wait = data[ind, 1] - data[ind, 0]
        ans  = np.mean(wait) * self.lam - np.mean(data[:, 3]) * self.rho

        self.assertAlmostEqual(ans, 0, 1)

    @unittest.skipIf(TRAVIS_TEST, reason)
    def test_LossQueue_blocking(self):

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        k   = np.random.randint(5, 15)
        scl = 1 / (mu * k)

        arr = lambda t : t + np.random.exponential(1/self.lam)
        ser = lambda t : t + np.random.gamma(k, scl)

        q2  = qt.LossQueue(nServers=nSe, arrival_f=arr, service_f=ser)
        q2.set_active()
        q2.simulate(n=100000)    # Burn in period

        nA0 = q2.nArrivals[1]
        nB0 = q2.nBlocked

        q2.simulate(n=250000)

        nA1 = q2.nArrivals[1]
        nB1 = q2.nBlocked

        a = self.lam / mu
        f = np.array([math.factorial(j) for j in range(nSe+1)])

        pois_pmf = np.exp(-a) * a**nSe / math.factorial(nSe)
        pois_cdf = np.sum(np.exp(-a) * a**np.arange(nSe+1) / f)
        p_block = (nB1 - nB0 + 0.0) / (nA1 - nA0)
        self.assertAlmostEqual(pois_pmf / pois_cdf, p_block, 2)



class TestRandomMeasure(unittest.TestCase):

    @unittest.skipIf(TRAVIS_TEST, reason)
    def test_poisson_random_measure(self):
        # This function tests to make sure the poisson_random_measure function
        # actually simulates a Poisson random measure. It does so looking for
        # Poisson random variables using a chi-squared test (testing the
        # composite null hypothesis). It does not look for independence of the
        # random variables.

        # This test should fail about
        
        rate  = lambda t: 0.5 + 4 * np.sin(np.pi * t / 12)**2
        arr_f = lambda t: qt.poisson_random_measure(rate, 4.5, t)

        nSamp = 15000
        nArr  = 1000
        arrival_times = np.zeros( (nSamp, nArr) )
        for k in range(nSamp):
            t = 0
            for j in range(nArr):
                t = arr_f(t)
                arrival_times[k, j] = t
                if t > 12 :
                    break
        
        mu1 = 5 * np.sum(rate(np.linspace(3, 8, 200))) / 200 # or 2*(5 + (sqrt(3) + 2) * 3/pi) + 2.5
        mu2 = 4 * np.sum(rate(np.linspace(8, 12, 200))) / 200 # or 2*(4 - 3*sqrt(3)/pi) + 2
        mus = [mu1, mu2]
        
        rv1 = np.sum(np.logical_and(3 < arrival_times, arrival_times < 8), axis=1)
        rv2 = np.sum(np.logical_and(8 < arrival_times, arrival_times < 12), axis=1)
        rvs = [rv1, rv2]
        df  = [max(rv1)+2, max(rv2)+2]
        
        Q = np.zeros( (max(df), len(rvs)) )
        
        for i, sample in enumerate(rvs):
            for k in range(df[i]-1):
                pi_hat  = nSamp * np.exp(-mus[i]) * mus[i]**k / math.factorial(k)
                Q[k, i] = (np.sum(sample == k) - pi_hat)**2 / pi_hat

            ans = np.array([math.factorial(j) for j in range(k+1)])
            pois_cdf = np.sum(np.exp(-mus[i]) * mus[i]**np.arange(k+1) / ans)
            Q[k+1, i] = nSamp * (1 - pois_cdf)
        
        Qs = np.sum(Q, axis=0)
        p  = np.array([1 - chi2_cdf(Qs[i], df[i]-2) for i in range(len(rvs))])
        self.assertTrue( (p > 0.1).any() )


if __name__ == '__main__':
    unittest.main()
