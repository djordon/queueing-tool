import numpy    as np
import scipy    as sp
import unittest

import queueing_tool as qt



def empirical_cdf0(x, z) :
    return np.sum(z <= x) / n

empirical_cdf = np.vectorize(empirical_cdf0, excluded={1})



class TestQueueServer(unittest.TestCase) :

    def setUp(self) :
        self.lam = np.random.randint(1,10)
        self.rho = np.random.uniform(0.5, 1)


    def test_markovian_queue(self) :
        """This function tests to make sure that the ``QueueServer`` class
        works correctly, by observing the departure times when in Equilibrium.
        It tests to see if the departure process of an ``M/M/k`` queue is
        exponentially distributed with mean thats identical to the mean of the
        input process. The test is a Kolmogorov-Smirnov test.

        Notes
        -----
        It relies on Burke's Theorem and the Dvoretzky-Kiefer-Wolfowitz
        Inequality. Burke's Theorem states that the output process is the input
        process in a (stable) markovian queue. The Dvoretzky-Kiefer-Wolfowitz
        inequality (D-K-W) states the following:

        .. math::

           \mathbb{P} \bigg\{ \sup_{x\in \R} |F(x) - \hat{F}_n(x)| > \epsilon \bigg\} 
                \leq 2\exp( - 2n \epsilon^2 ).

        See chapter 16 of [8]_ or [5]_ for a proof of Burke's Theorem and see
        [6]_ and [7]_ for proods of the D-K-W inequality.

        The :math:`\epsilon` chosen in here was such that

        .. math::

           2\exp( - 2n \epsilon^2 ) \approx 0.055

        References
        ----------
        .. [8] Harchol-Balter, Mor. *Performance Modeling and Design of\
               Computer Systems: Queueing Theory in Action*. Cambridge\
               University Press, 2013. ISBN: `9781107027503`_.

        .. [5] Burke, Paul J. "The output of a queuing system." *Operations\
               research* 4.6 (1956): 699-704. :doi:`10.1287/opre.4.6.699`

        .. [6] A. Dvoretzky, J. Kiefer, and J. Wolfowitz, "Asymptotic minimax\
               character of the sample distribution function and of the\
               classical multinomial estimator" Ann. Math. Statist. 27 (1956),\
               642–669.

        .. [7] P. Massart, The tight constant in the Dvoretzky-Kiefer-Wolfowitz\
               inequality, Ann. Probab. 18 (1990), no. 3, 1269–1283.

        .. _9781107027503: http://www.cambridge.org/us/9781107027503
        """
        lam = np.random.randint(1,10)
        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)

        arr = lambda t : t + np.random.exponential(1/lam)
        ser = lambda t : t + np.random.exponential(1/mu)

        q1  = qt.QueueServer(nServers=nSe, arrival_f=arr, service_f=ser)

        nSamp = 100

        n   = 5000
        ans = np.zeros(nSamp)

        for k in range(nSamp) :
            q1.set_active()
            q1.simulate(n=20000)    # Burn in period
            q1.collect_data = True
            q1.simulate(nD=n+1)
            dat = q1.fetch_data()
            dep = dat[:, 2][dat[:, 2] > 0]
            dep.sort()
            dep = dep[1:] - dep[:-1]
            nx  = 1001
            xx  = np.linspace(0, 10, nx)
            dif = np.abs(1 - np.exp(-xx * self.lam) - empirical_cdf(xx, dep))
            ans[k] = np.max(dif)
            q1.clear()

        min_eps = np.sqrt( np.log(6) / n )
        epsx    = np.linspace(min_eps, max([0.5, 3 * min_eps]), 100)
        dkw     = np.zeros(len(epsx), bool)

        for k, ep in enumerate(epsx) :
            dkw[k]  = np.sum(ans > ep) / nSamp <= 2 * np.exp( -2 * n * ep**2 )

        self.assertTrue( dkw.all() )


    def test_QueueNetwork_sorting(self) :

        g   = qt.generate_random_graph(300)
        qn  = qt.QueueNetwork(g)
        qn.agent_cap = 3000
        qn.initialize(50)
        qn.simulate(n=10000)

        nEvents = 1000
        ans = zeros(nEvents, bool)
        for k in range(nEvents) :
            net_times   = np.array([q.time for q in qn._queues])
            queue_times = [q.time for q in qn.edge2queue]
            queue_times.sort()
            while queue_times[-1] == np.infty :
                queue_times.pop()

            queue_times.sort(reverse=True)

            ans[k] = (queue_times == net_times).all()
            qn.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_LossQueue_blocking(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        k   = np.random.randint(5, 15)
        scl = 1 / (mu * k)

        arr = lambda t : t + np.random.exponential(1/lam)
        ser = lambda t : t + np.random.gamma(k, scl)

        q2  = qt.LossQueue(nServers=nSe, arrival_f=arr, service_f=ser)
        q2.set_active()
        q2.simulate(n=100000)    # Burn in period

        nA0 = q2.nArrivals[1]
        nB0 = q2.nBlocked

        q2.collect_data = True
        q2.simulate(n=250000)

        nA1 = q2.nArrivals[1]
        nB1 = q2.nBlocked

        a   = self.lam / mu

        erlangb = np.round(poisson.pmf(k , mu=a) / poisson.cdf(k, mu=a), 2)
        p_block = np.round((nB1 - nB0) / (nA1 - nA0), 2)

        self.assertTrue( np.isclose(erlangb, p_block) )


    def test_poisson_random_measure(self) :
        """This function tests to make sure the poisson_random_measure function
        actually simulates a Poisson random measure. It does so using a
        chi-squared test for the composite null hypothesis.
        """
        rate  = lambda t: 0.5 + 4 * np.sin(np.pi * t / 12)**2
        arr_f = lambda t: qt.poisson_random_measure(rate, 4.5, t)

        nSamp  = 5000
        nArr   = 100
        arrival_times = np.zeros( (nSamp, nArr) )
        for k in range(nSamp) :
            t = -10
            for j in range(nArr) :
                t = arr_f(t)
                arrival_times[k, j] = t

        mu1 = 3 * np.sum( rate(np.linspace(0, 3, 200)) ) / 200 # or 2*(3 - 6/pi) + 1.5
        mu2 = 6 * np.sum( rate(np.linspace(3, 9, 200)) ) / 200 # or 2*(6 + 2 * 6/pi) + 3
        mu3 = 3 * np.sum( rate(np.linspace(9, 12, 200)) ) / 200 # or 2*(3 - 6/pi) + 1.5
        mus = [mu1, mu2, mu3]

        rv1 = np.sum(np.logical_and(0 < arrival_times, arrival_times < 3), axis=1)
        rv2 = np.sum(np.logical_and(3 < arrival_times, arrival_times < 9), axis=1)
        rv3 = np.sum(np.logical_and(9 < arrival_times, arrival_times < 12), axis=1)
        rvs = [rv1, rv2, rv3]
        df  = [max(rv1)+2, max(rv2)+2, max(rv3)+2]

        Q   = np.zeros( (max(df), 3) )

        for i, sample in enumerate(rvs) :
            for k in range(df[i]) :
                pi_hat  = nSamp * poisson.pmf(k, mus[i])
                Q[k, i] = (np.sum(sample == k) - pi_hat)**2 / pi_hat
            
            Q[-1, i] = nSamp * (1 - poisson.cdf(k+1, mus[i]))

        Qs  = np.sum(Q[:,:], axis=0)
        p   = np.array([1 - stats.chi2.cdf(Qs[i], df[i]-2) for i in range(3)])
        self.assertTrue( (p > 0.1).all() )




if __name__ == '__main__':
    unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(TestQueueServer)
runner = unittest.TestQueueServer()
runner.run(suite)
