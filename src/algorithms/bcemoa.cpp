/* Copyright 2017-2018 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/bcemoa.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

bcemoa::bcemoa(unsigned gen, double cr, double eta_c, double m, double eta_m, unsigned seed)
    : nsga2(gen, cr, eta_c, m, eta_m, seed)
{
}

/// Algorithm evolve method
/**
 * Evolves the population for the requested number of generations.
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throw std::invalid_argument if pop.get_problem() is stochastic, single objective or has non linear constraints.
 * If \p int_dim is larger than the problem dimension. If the population size is smaller than 5 or not a multiple of
 * 4.
 */
population bcemoa::evolve(population pop) const
{
    // Call evolve from parent class
    return nsga2::evolve(pop);
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void bcemoa::set_seed(unsigned seed)
{
    nsga2::set_seed(seed);
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string bcemoa::get_extra_info() const
{
    std::ostringstream ss;
    // Get info from parent class
    stream(ss, nsga2::get_extra_info());
    // FIXME: Add BCEMOA specific info
    return ss.str();
}

/// Object serialization
/**
 * This method will save/load \p this into the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
 */
template <typename Archive>
void bcemoa::serialize(Archive &ar, unsigned x)
{
    nsga2::serialize(ar, x);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::bcemoa)
