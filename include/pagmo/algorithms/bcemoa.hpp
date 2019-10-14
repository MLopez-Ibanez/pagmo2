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

#ifndef PAGMO_ALGORITHMS_BCEMOA_HPP
#define PAGMO_ALGORITHMS_BCEMOA_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithms/learner.hpp>
#include <pagmo/algorithms/machineDM.hpp>
#include <pagmo/algorithms/nsga2.hpp>

namespace pagmo
{
/// BCEMOA
/**
 * DOCUMENT ME !
 */
class PAGMO_DLL_PUBLIC bcemoa : public nsga2
{

public:
    /// Constructor
    /**
     * Constructs the BCEMOA user defined algorithm.
     *
     * @param[in] gen Number of generations to evolve.
     * @param[in] cr Crossover probability.
     * @param[in] eta_c Distribution index for crossover.
     * @param[in] m Mutation probability.
     * @param[in] eta_m Distribution index for mutation.
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
     * [1,100[ or \p eta_m is not in [1,100[.
     */
    // FIXME: Cannot add empty parameter machineDM dm

    bcemoa(svm &ml, unsigned gen1 = 1u, unsigned geni = 10u, int maxInteractions = 20, int n_of_evals = 5,
           double cr = 0.95, double eta_c = 10., double m = 0.01, double eta_m = 50.,
           unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population bc_evolve(population &);

    // population evolvedm(machineDM &dm, population);
    // Algorithm evolve based on preference information
    population evolvei(population &);

    // FIXME: Report to pagmo that if we don't duplicate this, we get
    /* bcemoa.o: In function `pagmo::detail::algo_inner<pagmo::bcemoa>::set_seed(unsigned int)':
bcemoa.cpp:(.text._ZN5pagmo6detail10algo_innerINS_6bcemoaEE8set_seedEj[_ZN5pagmo6detail10algo_innerINS_6bcemoaEE8set_seedEj]+0x5):
undefined reference to `pagmo::bcemoa::set_seed(unsigned int)'
    */
    // Sets the seed
    void set_seed(unsigned);

    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "BCEMOA:";
    }
    std::string get_extra_info() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

    // private:
    //    bcemoa

    unsigned m_geni;
    svm ml;
    int maxInteractions;
    int n_of_evals;
};

} // namespace pagmo
// M: I noticed it had not been working in machinDM as well. I tried to figure it out and I failed
// PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::bcemoa : nsga2)

#endif
