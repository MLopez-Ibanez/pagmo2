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
#include <pagmo/algorithms/machineDM.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

bcemoa::bcemoa(unsigned gen1, unsigned geni, double cr, double eta_c, double m, double eta_m, unsigned seed)
    : nsga2(gen1, cr, eta_c, m, eta_m, seed), m_geni(geni)
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
    // Call evolve from parent class (NSGA-II) for gen1
    pop = nsga2::evolve(pop);
    // Call interactive evolve of BCEMOA for geni
    return evolvei(pop);
}

population bcemoa::evolvei(population pop) const
   {
       // We store some useful variables
       const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                             // allowed
       auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
       auto NP = pop.size();

       auto fevals0 = prob.get_fevals(); // discount for the fevals already made
       unsigned int count = 1u;          // regulates the screen output

       // PREAMBLE-------------------------------------------------------------------------------------------------
       // We start by checking that the problem is suitable for this
       // particular algorithm.
       if (detail::some_bound_is_equal(prob)) {
           pagmo_throw(
               std::invalid_argument,
               get_name()
                   + " cannot work on problems having a lower bound equal to an upper bound. Check your bounds.");
       }
       if (prob.is_stochastic()) {
           pagmo_throw(std::invalid_argument,
                       "The problem appears to be stochastic " + get_name() + " cannot deal with it");
       }
       if (prob.get_nc() != 0u) {
           pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                  + get_name() + " cannot deal with them.");
       }
       if (prob.get_nf() < 2u) {
           pagmo_throw(std::invalid_argument,
                       "This is a multiobjective algortihm, while number of objectives detected in " + prob.get_name()
                           + " is " + std::to_string(prob.get_nf()));
       }
       if (NP < 5u || (NP % 4 != 0u)) {
           pagmo_throw(std::invalid_argument,
                       "for NSGA-II at least 5 individuals in the population are needed and the "
                       "population size must be a multiple of 4. Detected input population size is: "
                           + std::to_string(NP));
       }
       // ---------------------------------------------------------------------------------------------------------

       // No throws, all valid: we clear the logs
       m_log.clear();

       // Declarations
       std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
       vector_double::size_type parent1_idx, parent2_idx;
       vector_double child1(dim), child2(dim);

       std::iota(shuffle1.begin(), shuffle1.end(), 0u);
       std::iota(shuffle2.begin(), shuffle2.end(), 0u);

       // Main NSGA-II loop
       for (decltype(m_gen) gen = 1u; gen <= m_geni; gen++) {
           // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
           if (m_verbosity > 0u) {
               // Every m_verbosity generations print a log line
               if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                   // We compute the ideal point
                   vector_double ideal_point = ideal(pop.get_f());
                   // Every 50 lines print the column names
                   if (count % 50u == 1u) {
                       print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                       for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                           if (i >= 5u) {
                               print(std::setw(15), "... :");
                               break;
                           }
                           print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                       }
                       print('\n');
                   }
                   print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);
                   for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                       if (i >= 5u) {
                           break;
                       }
                       print(std::setw(15), ideal_point[i]);
                   }
                   print('\n');
                   ++count;
                   // Logs
                   m_log.emplace_back(gen, prob.get_fevals() - fevals0, ideal_point);
               }
           }

           // At each generation we make a copy of the population into popnew
           population popnew(pop);

           // We create some pseudo-random permutation of the poulation indexes
           std::shuffle(shuffle1.begin(), shuffle1.end(), m_e);
           std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);

           // 1 - We compute crowding distance and non dominated rank for the current population
           auto fnds_res = fast_non_dominated_sorting(pop.get_f());
           auto ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
           vector_double pop_cd(NP);         // We use preference instead of crowding distances of the whole population
           auto ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
           vector_double v;
     for (const auto &front_idxs : ndf) {
               std::vector<vector_double> front;
               for (auto idx : front_idxs) {

                   v=pop.get_f()[idx];
                   //pop_cd[idx] = accumulate(v.begin(), v.end(), 0.0) / v.size();
                   vector_double w(v.size(), 1.0 / v.size());
                   linear_value_function vf{w};
                   pop_cd[idx] =   vf.value(v);//linear_value_function::value(v);//dm.linear_utility(v, w, ideal_point)
               }
           }

           // 3 - We then loop thorugh all individuals with increment 4 to select two pairs of parents that will
           // each create 2 new offspring
           for (decltype(NP) i = 0u; i < NP; i += 4) {
               // We create two offsprings using the shuffled list 1
               parent1_idx = tournament_selection(shuffle1[i], shuffle1[i + 1], ndr, pop_cd);
               parent2_idx = tournament_selection(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd);
               crossover(child1, child2, parent1_idx, parent2_idx, pop);
               mutate(child1, pop);
               mutate(child2, pop);
               // we use prob to evaluate the fitness so
               // that its feval counter is correctly updated
               auto f1 = prob.fitness(child1);
               auto f2 = prob.fitness(child2);
               popnew.push_back(child1, f1);
               popnew.push_back(child2, f2);

               // We repeat with the shuffled list 2
               parent1_idx = tournament_selection(shuffle2[i], shuffle2[i + 1], ndr, pop_cd);
               parent2_idx = tournament_selection(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd);
               crossover(child1, child2, parent1_idx, parent2_idx, pop);
               mutate(child1, pop);
               mutate(child2, pop);
               // we use prob to evaluate the fitness so
               // that its feval counter is correctly updated
               f1 = prob.fitness(child1);
               f2 = prob.fitness(child2);
               popnew.push_back(child1, f1);
               popnew.push_back(child2, f2);
           } // popnew now contains 2NP individuals

           // This method returns the sorted N best individuals in the population according to the crowded comparison
           // operator
           //shouldn't we sort individuals based on preference function???
           best_idx = select_best_N_mo(popnew.get_f(), NP);
           // We insert into the population
           for (population::size_type i = 0; i < NP; ++i) {
               pop.set_xf(i, popnew.get_x()[best_idx[i]], popnew.get_f()[best_idx[i]]);
           }
       } // end of main NSGAII loop
       return pop;
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
