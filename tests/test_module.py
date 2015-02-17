import numpy    as np
import unittest

import queueing_tool as qt



def empirical_cdf0(x, z) :
    return np.sum(z <= x) / n

empirical_cdf = np.vectorize(empirical_cdf0, excluded={1})



class TestQueueServer(unittest.TestCase) :

    def setUp(self) :
        lam = np.random.randint(1,10)
        mu  = lam * np.random.uniform()
        k   = np.random.randint(5, 15)
        scl = mu / k
        nSr = np.random.randint(5, 10)

        arr_f   = lambda t : t + np.random.exponential(1/lam)
        ser_f1  = lambda t : t + np.random.exponential(1/mu)
        ser_f2  = lambda t : t + np.random.gamma(k, scale)

        self.lam = lam
        self.mu  = mu
        self.q1  = qt.QueueServer(nServers=1, arrival_f=arr_f, service_f=ser_f1)
        self.q2  = qt.LossQueue(nServers=nSr, arrival_f=arr_f, service_f=ser_f2)

    def test_markovian_queue(self) :
        """This function tests to make sure that the ``QueueServer`` class works
        correctly, by observing the departure times when in Equilibrium. 

        Notes
        -----
        It relies on Burke's Theorem and the Dvoretzky-Kiefer-Wolfowitz
        Inequality. Burke's Theorem states that the output process is the input
        process in a (stable) markovian queue. The Dvoretzky-Kiefer-Wolfowitz
        inequality (D-K-W) states the following:

        .. math::

            \mathbb{P} \bigg\{ \sup_{x\in \R} |F(x) - \hat{F}_n(x)| > \epsilon \bigg\} 
                \leq 2\exp( - 2n \epsilon^2 ).

        See chapter 16 of [3]_ or [5]_ for a proof of Burke's Theorem and see
        [6]_ and [7]_ for proods of the D-K-W inequality.

        References
        ----------
        .. [3] Harchol-Balter, Mor. *Performance Modeling and Design of\
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
        nS  = 100
        n   = 5000
        ans = np.zeros(nS)

        for k in range(nS) :
            self.q1.simulate(n=20000)    # Burn in period
            self.q1.set_active()
            self.q1.collect_data = True
            self.q1.simulate(nD=n+1)
            dat = self.q1.fetch_data()
            dep = dat[:, 2][dat[:, 2] > 0]
            dep.sort()
            dep = dep[1:] - dep[:-1]
            nx  = 1001
            xx  = np.linspace(0, 10, nx)
            dif = np.abs(1 - np.exp(-xx * self.lam) - empirical_cdf(xx, dep))
            ans[k] = np.max(dif)
            self.q1.clear()

        min_eps = np.sqrt( np.log(6) / n )
        epsx    = np.linspace(min_eps, max([0.5, 3 * min_eps]), 100)
        dkw     = np.zeros(len(epsx), bool)

        for k, ep in enumerate(epsx) :
            dkw[k]  = np.sum(ans > ep) / nS <= 2 * np.exp( -2 * n * ep**2 )

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

if __name__ == '__main__':
unittest.main()
suite = unittest.TestLoader().loadTestsFromTestCase(TestQueueServer)
runner = unittest.TestQueueServer()
runner.run(suite)
