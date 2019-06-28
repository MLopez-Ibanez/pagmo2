#ifndef PAGMO_ALGORITHMS_MACHINEDM_HPP
#define PAGMO_ALGORITHMS_MACHINEDM_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
class PAGMO_DLL_PUBLIC value_function
{
public:

    virtual double value(const std::vector<double> &) const = 0;
    std::vector<double> weights;
    std::vector<double> ideal_point;

protected:
    value_function(std::vector<double> w,std::vector<double> ip={0,0})
        : weights(w),ideal_point(ip)
    {};
};

class PAGMO_DLL_PUBLIC linear_value_function : public value_function
{
public:

    linear_value_function(std::vector<double> w)
        : value_function(w)
    {};
    double value(const std::vector<double> &) const;
};
class PAGMO_DLL_PUBLIC quadratic_value_function : public value_function
{
public:
quadratic_value_function ( std::vector<double> w, std::vector<double> ip)
        : value_function(w,ip)
{};
double value(const std::vector<double> &) const;
};
/// Machine Decision Maker
/**
 * FIXME: DOCUMENT ME !
 */
class PAGMO_DLL_PUBLIC machineDM
{
public:
    /// Constructor
    /**
     * FIXME: Update documentation: Constructs a Machine Decision Maker
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
    machineDM(problem &prob, value_function &pref,
              unsigned seed = pagmo::random_device::next())
        : prob(prob), pref(pref)
    {};

    /**
     * Evaluate fitness (objective vector) according to decision maker.
     *
     *
     **/
    vector_double fitness(const vector_double &) const;

    /**
     * Solution value according to DM.
     *
     *
     **/
    double value(const vector_double &) const;

    /**
     * True solution value according to DM's preference function.
     *
     *
     **/
    double true_value(const vector_double &) const;

    std::vector<size_t> rank(const pagmo::population &pop) const;

    template <typename Archive> void serialize(Archive &ar, unsigned);

    value_function & pref;
    problem &prob;
};

} // namespace pagmo

// FIXME: Not working
//PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::machineDM)

#endif
