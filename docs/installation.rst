Installation
============

The queueing-tool module provides a framework for creating, simulating, and
visualizing queueing networks. The network visualizations are handled by graph-tool,
which is required in order to use this package. You will also need numpy (and
scipy to run tests). The package works with python versions 2.7 and 3.2 or greater.

Installation instructions for numpy can be found in the `numpy docs`_\. There
are installable binary packages available for Windows, Mac OS X, and many
distributions of Linux. For graph-tool, there are precompiled `packages`_ made
for Debian & Ubuntu, as well as `Macports`_ portfiles for those using Mac OS X.
Unfortunately, there does not seem to be any options available for Windows users.

You can obtain queueing-tool from it's `github repository`_. If you follow
the link you can download a zip file of the package. Alternatively, you can clone
it to your desktop using git::

    $ git clone https://github.com/djordon/queueing-tool

Once you have the package locally on your computer, change directories to the
queueing-tool directory in your terminal and install using::

    $ sudo python3 setup.py install

Alternatively, you could install it locally with::

    $ python3 setup.py install --user

Note that this package was tested using numpy version 1.9.1 and graph-tool 
versions 2.2.31.

.. _numpy docs: http://docs.scipy.org/doc/numpy/user/install.html
.. _packages: http://graph-tool.skewed.de/download#packages
.. _Macports: http://graph-tool.skewed.de/download#macos
.. _github repository: https://github.com/djordon/queueing-tool
