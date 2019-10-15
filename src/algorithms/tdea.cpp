

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
#include <pagmo/algorithms/tdea.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

tdea::tdea(unsigned gen, double cr, double eta_c, double m, double eta_m, double tau, unsigned seed)
    : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_eta_m(eta_m), m_e(seed), m_tau(tau), m_seed(seed), m_verbosity(0u)
{
    if (cr >= 1. || cr < 0.) {
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1[ range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (m < 0. || m > 1.) {
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the [0,1] range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (eta_c < 1. || eta_c > 100.) {
        pagmo_throw(std::invalid_argument, "The distribution index for crossover must be in [1, 100], while a value of "
                                               + std::to_string(eta_c) + " was detected");
    }
    if (eta_m < 1. || eta_m > 100.) {
        pagmo_throw(std::invalid_argument, "The distribution index for mutation must be in [1, 100], while a value of "
                                               + std::to_string(eta_m) + " was detected");
    }
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
population tdea::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
    auto NP = pop.size();

    auto fevals0 = prob.get_fevals();                // discount for the fevals already made
    unsigned count = 1u, nx = pop.get_x()[1].size(); // regulates the screen output
    bool dominated;

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument,
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
        pagmo_throw(std::invalid_argument, "This is a multiobjective algortihm, while number of objectives detected in "
                                               + prob.get_name() + " is " + std::to_string(prob.get_nf()));
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
    population archive{prob};
    std::vector<int> best_idx(NP), shuffle1(NP);
    int parent1_idx, parent2_idx;
    vector_double child1(dim), child2(dim), parent1, parent2;

    std::iota(shuffle1.begin(), shuffle1.end(), 0u);

    // Main NSGA-II loop
    for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
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

        // At each generation we make a copy of the archieve into popnew
        // opulation popnew(archieve);

        // We create some pseudo-random permutation of the poulation indexes

        std::shuffle(shuffle1.begin(), shuffle1.end(), m_e);
        std::vector<vector_double> obf = pop.get_f();

        // Two individuals are selected randomly from the population; the prioritiy is given to the one that dominates,
        // in none of them dominate oother one of them is selected randomly

        if (pareto_dominance(obf[shuffle1[0]], obf[shuffle1[1]])) {
            parent1_idx = shuffle1[0];
        } else if (pareto_dominance(obf[shuffle1[1]], obf[shuffle1[0]]) || ((double)rand() / (RAND_MAX)) > 0.5) {
            parent1_idx = shuffle1[1];
        } else {
            parent1_idx = shuffle1[0];
        }
        parent1 = pop.get_x()[parent1_idx];

        // parent 2 is selected randomly from Archive, if Archiive is emmpy one individual is randomly selected from the
        // population
        if (archive.size() < 2) {
            parent2 = pop.get_x()[shuffle1[2]];
        } else {
            // Second shuffle has the size of archive
            std::vector<int> shuffle2(archive.size());
            std::iota(shuffle2.begin(), shuffle2.end(), 0u); // M: This line may be deleted
            std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);
            parent2_idx = shuffle2[1];
            parent2 = archive.get_x()[parent2_idx];
        }

        crossover(child1, child2, parent1, parent2, pop);
        mutate(child1, pop);

        // Inserting child into population
        // checking if the child is dominated by any individuals in pop
        dominated = false;
        std::vector<int> dominated_list;

        vector_double child_obf = prob.fitness(child1);

        for (int i = 0; i < NP; i++) {
            if (pareto_dominance(obf[i], child_obf)) {
                dominated = true;
                break;
            }
        }
        if (dominated == true) continue;
        for (int i = 0; i < NP; i++) {
            if (pareto_dominance(child_obf, obf[i])) {
                dominated_list.push_back(i);
            }
        }

        // the child replaces one of the dominated individuals in population, if it does not dominate any individual it
        // would replace one randomly
        if (dominated_list.size() > 1) {
            pop.set_xf(dominated_list[ceil(rand() / (RAND_MAX)) * dominated_list.size()], child1, child_obf);
        } else {
            pop.set_xf(ceil(rand() / (RAND_MAX)) * NP, child1, child_obf);
        }

        // inserting child into Archive
        if (archive.size() < 2) {
            archive.push_back(child1, child_obf);
            continue;
        }
        obf = archive.get_f();
        dominated = false;
        for (int i = 0; i < obf.size(); i++) {
            if (pareto_dominance(obf[i], child_obf)) {
                dominated = true;
                break;
            }
        }
        if (dominated == true) continue;

        // calculating the rectilinear distance of child to each individual in archive
        double dc, min_dc = std::numeric_limits<double>::max();
        int argmin;
        std::vector<vector_double> archive_x = archive.get_x();
        for (int i = 0; i < archive.size(); i++) {
            dc = 0;
            for (int j = 0; j < nx; j++) {
                dc += abs(archive_x[i][j] - child1[j]);
            }
            if (dc < min_dc) {
                min_dc = dc;
                argmin = i;
            }
        }

        // Finding the maximum scaled absolute objective difference between child and archive[i]
        double max_dis = 0;
        for (int j = 0; j < nx; j++) {
            if (archive_x[argmin][j] - child1[j] > m_tau) {
                archive.push_back(child1, child_obf);
            }
        }

        // This method returns the sorted N best individuals in the population according to the crowded comparison
        // operator
        // best_idx = select_best_N_mo(archive.get_f(), NP);
        // // We insert into the population
        // for (population::size_type i = 0; i < NP; ++i) {
        //     pop.set_xf(i, archive.get_x()[best_idx[i]], archive.get_f()[best_idx[i]]);
        // }
    } // end of main NSGAII loop
    return archive;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void tdea::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string tdea::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tCrossover probability: ", m_cr);
    stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
    stream(ss, "\n\tMutation probability: ", m_m);
    stream(ss, "\n\tDistribution index for mutation: ", m_eta_m);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    return ss.str();
}
//
// vector_double::size_type tdea::tournament_selection(vector_double::size_type idx1, vector_double::size_type idx2)
// const
// {
//     if (pareto_dominance(idx1,idx2)) return idx1;
//     if (non_domination_rank[idx1] > non_domination_rank[idx2]) return idx2;
//     if (crowding_d[idx1] > crowding_d[idx2]) return idx1;
//     if (crowding_d[idx1] < crowding_d[idx2]) return idx2;
//     std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)
//     return ((drng(m_e) > 0.5) ? idx1 : idx2);
// }

void tdea::crossover(vector_double &child1, vector_double &child2, vector_double parent1, vector_double parent2,
                     const pagmo::population &pop) const
{
    // Decision vector dimensions
    auto D = pop.get_problem().get_nx();
    auto Di = pop.get_problem().get_nix();
    auto Dc = pop.get_problem().get_ncx();
    // Problem bounds
    const auto bounds = pop.get_problem().get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;

    // declarations
    double y1, y2, yl, yu, rand01, beta, alpha, betaq, c1, c2;
    vector_double::size_type site1, site2;
    // Initialize the child decision vectors
    child1 = parent1;
    child2 = parent2;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

    // This implements a Simulated Binary Crossover SBX and applies it to the non integer part of the decision
    // vector
    if (drng(m_e) <= m_cr) {
        for (decltype(Dc) i = 0u; i < Dc; i++) {
            if ((drng(m_e) <= 0.5) && (std::abs(parent1[i] - parent2[i])) > 1e-14 && lb[i] != ub[i]) {
                if (parent1[i] < parent2[i]) {
                    y1 = parent1[i];
                    y2 = parent2[i];
                } else {
                    y1 = parent2[i];
                    y2 = parent1[i];
                }
                yl = lb[i];
                yu = ub[i];
                rand01 = drng(m_e);
                beta = 1. + (2. * (y1 - yl) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                }
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                beta = 1. + (2. * (yu - y2) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                }
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                if (c1 < lb[i]) c1 = lb[i];
                if (c2 < lb[i]) c2 = lb[i];
                if (c1 > ub[i]) c1 = ub[i];
                if (c2 > ub[i]) c2 = ub[i];
                if (drng(m_e) <= .5) {
                    child1[i] = c1;
                    child2[i] = c2;
                } else {
                    child1[i] = c2;
                    child2[i] = c1;
                }
            }
        }
    }
    // This implements two-point binary crossover and applies it to the integer part of the chromosome
    for (decltype(Dc) i = Dc; i < D; ++i) {
        // in this loop we are sure Di is at least 1
        std::uniform_int_distribution<vector_double::size_type> ra_num(0, Di - 1u);
        if (drng(m_e) <= m_cr) {
            site1 = ra_num(m_e);
            site2 = ra_num(m_e);
            if (site1 > site2) {
                std::swap(site1, site2);
            }
            for (decltype(site1) j = 0u; j < site1; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
            for (decltype(site2) j = site1; j < site2; ++j) {
                child1[j] = parent2[j];
                child2[j] = parent1[j];
            }
            for (decltype(Di) j = site2; j < Di; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
        } else {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    }
}

void tdea::mutate(vector_double &child, const pagmo::population &pop) const
{
    // Decision vector dimensions
    auto D = pop.get_problem().get_nx();
    auto Dc = pop.get_problem().get_ncx();
    // Problem bounds
    const auto bounds = pop.get_problem().get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // declarations
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

    // This implements the real polinomial mutation and applies it to the non integer part of the decision vector
    for (decltype(Dc) j = 0u; j < Dc; ++j) {
        if (drng(m_e) <= m_m && lb[j] != ub[j]) {
            y = child[j];
            yl = lb[j];
            yu = ub[j];
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = drng(m_e);
            mut_pow = 1. / (m_eta_m + 1.);
            if (rnd <= 0.5) {
                xy = 1. - delta1;
                val = 2. * rnd + (1. - 2. * rnd) * (std::pow(xy, (m_eta_m + 1.)));
                deltaq = std::pow(val, mut_pow) - 1.;
            } else {
                xy = 1. - delta2;
                val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * (std::pow(xy, (m_eta_m + 1.)));
                deltaq = 1. - (std::pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            if (y < yl) y = yl;
            if (y > yu) y = yu;
            child[j] = y;
        }
    }

    // This implements the integer mutation for an individual
    for (decltype(D) j = Dc; j < D; ++j) {
        if (drng(m_e) < m_m) {
            // We need to draw a random integer in [lb, ub].
            auto mutated = uniform_integral_from_range(lb[j], ub[j], m_e);
            child[j] = mutated;
        }
    }
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::tdea)
