# Changelog

All notable changes to this project will be documented in this file.

## [1.2.5] - 2022-10-08

### Bug Fixes

- Address queue server bugs when changing the number of servers ([#77](https://github.com/djordon/queueing-tool/pull/77))
  * Make sure we correctly handle queueing when the number of servers changes for the `QueueServer` (fixes [#64](https://github.com/djordon/queueing-tool/issues/64)).
  * Update the requirements file for building the documentation.
  * Update the javascript files included in the documentation.

- Address issues related to newer versions of dependencies ([#75](https://github.com/djordon/queueing-tool/pull/75))
  * Move CI over from travis-ci to Github actions. Test python versions 3.6-3.10.
  * Use poetry for package management.
  * Make sure queueing-tool works with updated dependencies (fixes [#74](https://github.com/djordon/queueing-tool/issues/74)).
  * Address warnings after python version and dependency updates.
  * Use the `collections.deque.clear()` method in the `QueueNetwork.clear()` function, (addresses [#68](https://github.com/djordon/queueing-tool/issues/68)).


### Build

- Use pyproject.toml for all build and test related configuration ([#79](https://github.com/djordon/queueing-tool/pull/79))
  * Move build metadata from `setup.py` to `pyproject.toml`.
  * Move pytest settings from `setup.cfg` to `pyproject.toml`.
  * Use `importlib.metadata` for setting the package version.
  * Remove the version file.


### Documentation

- Update documentation for QueueServer attributes ([#78](https://github.com/djordon/queueing-tool/pull/78))
  * Clarified the documentation of `QueueServer.num_arrivals` and `QueueServer.num_system` (addresses [#65](https://github.com/djordon/queueing-tool/issues/65)).
  * Setup a `CHANGELOG.md` file.



## [1.2.4] - 2019-11-10

### Changed

* Networkx nodes update ([#61](https://github.com/djordon/queueing-tool/pull/61))
  * Use nodes instead of node, the latter is removed as of networkx 2.4


### Documentation

- Update readthedocs v2 ([#59](https://github.com/djordon/queueing-tool/pull/59))
  * Use python 3.6 with RTD
  * Change the copy in the README
- Update readthedocs ([#58](https://github.com/djordon/queueing-tool/pull/58))
  * Update the installlation documentation
  * Update alabaster version for docs


## [1.2.3] - 2019-04-06

### Features

* Updates for python 3.6 and 3.7 ([#56](https://github.com/djordon/queueing-tool/pull/56))
  * Update the supported versions of python
  * Various non-breaking changes to ensure consistent results across versions of python


## [1.2.2] - 2018-12-22

### Features

- Add support for matplotlib 2 and 3 ([#52](https://github.com/djordon/queueing-tool/pull/52))
  * Add matrix of tests for matplotlib versions
  * Remove verbal support for python version 3.3


## [1.2.1] - 2017-10-18

### Changed

- [**breaking**] Better testing ([#45](https://github.com/djordon/queueing-tool/pull/45))
  * Code quality cleanup
  * Linted the files. Some docstring copy edits. Removed deepcopy implementation.
  * Remove unused imports
  * Renamed am image file, and added another image file for testing matplotlib 2.x
  * Updated the graph object to work with networkx 1.x and 2.x


## [1.2.0] - 2016-09-11

### Changed

- [**breaking**] QueueServer fixes ([#44](https://github.com/djordon/queueing-tool/pull/44))
  * Removed lowerCammelCase variables.
  * Moved `QueueServer` time updates into a `_update_time` functoin.
  * Minor `QueueNetwork.animation` code clean-up.

- Added some shields, switched to using pytest ([#43](https://github.com/djordon/queueing-tool/pull/43))


## [1.1.1] - 2016-07-02

### Changed

- [**breaking**] Pythonic names ([#39](https://github.com/djordon/queueing-tool/pull/39))
  * Updated un-pythonic attributions and parameters
  * Removed lambda functions in examples and docstrings
  * Updated tests to use more specific checks


## [1.1.0] - 2016-06-04

### Changed

- [**breaking**] Modify random measure ([#36](https://github.com/djordon/queueing-tool/pull/36))
  * Changed order of arguments for poisson_random_measure
  * Added priority queue to available package objects
  * PriorityQueue c changes
  * Updated the version, added PyPi info to README


## [1.0.4] - 2016-05-08

### Documentation

- Fix rtfd ([#34](https://github.com/djordon/queueing-tool/pull/34))
  * Edited `conf.py` so that it can be installed in python 2
  * Add documentation packages to the requirements
  * Fixed annoying readthedocs bug
- Code quality changes ([#33](https://github.com/djordon/queueing-tool/pull/33))
- Minor code quality changes ([#32](https://github.com/djordon/queueing-tool/pull/32))



## [1.0.3] - 2016-04-10

### Documentation

- Doc changes ([#30](https://github.com/djordon/queueing-tool/pull/30))


## [1.0.2] - 2016-04-09

### Changed

- Better testing ([#29](https://github.com/djordon/queueing-tool/pull/29))
  * Relaxed the numpy and networkx dependency requirements.
  * Added coloring parameter to QueueServer.


## [0.1.2] - 2015-03-30

### Changed

- Fixed a sorting bug that occurred under Blocking After Service. It
was solved by adding a small random number to a blocked agent's
next time. This implies that the first agent blocked will not
necessarily be the next agent to enter the queue when it is no longer
at capacity.

Improved efficiency of the GreedyAgent routing; it's now about 10
times faster.

- Fixed installation typo in the docs.
- Updated installation docs.


## [0.1.1] - 2015-03-22

### Changed

- [**breaking**] Modified various functions. Better simulation for QueueServer.

  **QueueNetwork**
  - Changed the name of the `collect_data` method to
    `start_collecting_data`.
  - Changed the `time` attribute to tract the time of next event.
  - Added a `current_time` attribute.

  **QueueServer**
  - Changed the `simulate` method to exit when no new events
    are scheduled.
  - The `fetch_data` method now sorts the entries by arrival time.

- Made blocking attribute undeletable


## [0.1.0] - 2015-03-03

### Changed

- Initial release

<!-- generated by git-cliff -->
