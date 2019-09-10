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

#include <boost/range/combine.hpp>
#include <boost/foreach.hpp>

#include <pagmo/algorithm.hpp>
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

double linear_value_function::value(const std::vector<double> & obj) const
{
    double value = 0.;
    // C++17:
    /*   for(auto const& [ti,tc] : boost::combine(v, l)) {
         std::cout << '(' << ti << ',' << tv << ')' << '\n';
         }
    */
    double w, o;
    BOOST_FOREACH(boost::tie(w,o), boost::combine(this->weights, obj)) {
        value += w * o;
    }
    return value;
}
double quadratic_value_function::value(const std::vector<double> & obj) const
{
double value = 0.;
double w, ip, o;
BOOST_FOREACH(boost::tie(w, ip, o) , boost::combine(this->weights, this->ideal_point, obj)) {
    value += pow(w * (o-ip),2);
}
return value;

}



vector_double
machineDM::fitness(const vector_double & solution) const
{
    // FIXME: Apply biases
    vector_double f = prob.fitness(solution);
    return f;
}

double
machineDM::value(const vector_double & solution) const
{
    vector_double f = this->fitness(solution);
    // FIXME: Apply biases
    return pref.value(f);
}

double
machineDM::true_value(const vector_double & solution) const
{
    vector_double f = prob.fitness(solution);
    return pref.value(f);
}

std::vector<size_t>
machineDM::rank(const pagmo::population &pop) const
{
    std::vector<size_t> ranks;
    // FIXME
    return ranks;
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
void machineDM::serialize(Archive &ar, unsigned x)
{
    // FIXME
}

} // namespace pagmo

// FIXME: not working
// PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::machineDM)