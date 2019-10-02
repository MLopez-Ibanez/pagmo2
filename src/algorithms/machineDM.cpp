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

#include <boost/foreach.hpp>
#include <boost/range/combine.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/machineDM.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

#define NI 1  // No interaction
#define ITV 2 // Interaction with true value
#define IMV   // Interaction with modified value

namespace pagmo
{

double linear_value_function::value(const std::vector<double> &obj) const
{
    double value = 0.;
    // C++17:
    /*   for(auto const& [ti,tc] : boost::combine(v, l)) {
         std::cout << '(' << ti << ',' << tv << ')' << '\n';
         }
    */
    double w, o;
    BOOST_FOREACH (boost::tie(w, o), boost::combine(this->weights, obj)) {
        value += w * o;
    }
    return value;
}
double quadratic_value_function::value(const std::vector<double> &obj) const
{
    double value = 0.;
    double w, ip, o;
    BOOST_FOREACH (boost::tie(w, ip, o), boost::combine(this->weights, this->ideal_point, obj)) {
        value += pow(w * (o - ip), 2);
    }
    return value;
}

double tchebycheff_value_function::value(const std::vector<double> &obj) const
{
    double temp;
    double w, ip, o;
    double value = 0.;
    BOOST_FOREACH (boost::tie(w, ip, o), boost::combine(this->weights, this->ideal_point, obj)) {
        temp = w * std::abs(o - ip);
        if (temp > value) {
            value = temp;
        }
    }
    return value;
}
/* Select a subset of q criteria from m true criteria with probability
   proportional to their weights.  */
std::vector<int> machineDM::select_criteria_subset()
{
    vector_double w = this->pref.weights;
    std::vector<int> c(w.size());
    double sum = accumulate(w.begin(), w.end(), 0);
    vector_double probs(w.size());
    for (size_t i = 0; i < w.size(); i++) {
        probs[i] = w[i] / sum; // M: instead of ranking the elements based on weights, I directly use weights to
                               // calculate the prob. There is no need to rank them I think
    }
    int k;
    for (size_t i = 0; i < w.size(); i++) {
        k = roulette_wheel(probs);
        c[i] = k;
        probs[k] = 0.0;
    }
    return c;
}
vector_double machineDM::get_weights()
{
    return this->pref.weights;
}
// This roulette_wheel function returns an index of an element based on the
// probabilities constructed based on the values of the passed vector
int machineDM::roulette_wheel(vector_double &v)
{
    double sum = 0.;
    double rndNumber;
    int i;
    for (i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    srand((unsigned)time(NULL)); // M: I'm not sure if this line randomizes the seed or if we should randomize it at all
    rndNumber = ((double)rand() / (double)RAND_MAX) * sum;
    for (i = 0; i < v.size(); i++) {
        if (v[i] >= rndNumber) {
            return i;
        }
    }
}
vector_double machineDM::modify_criteria(
    vector_double &obf,
    const std::vector<int> &c) // M: I have assumed that tau in previous vrsion is ideal point. thus I have changed it
{
    assert(gamma >= 0);
    assert(gamma < 1);
    int m = obf.size();
    vector_double ip = this->pref.ideal_point;
    vector_double zhat(m);
    for (int k = 1; k < q; k += 2) {
        assert(k < m);
        assert(c[k - 1] >= 0);
        assert(c[k - 1] < m);
        assert(c[k] >= 0);
        assert(c[k] < m);
        zhat[k - 1] = (1.0 - gamma) * obf[c[k - 1]] + gamma * obf[c[k]];
        zhat[k] = (1.0 - gamma) * obf[c[k]] + gamma * obf[c[k - 1]];
    }
    // if q is odd.
    if (q % 2) zhat[q - 1] = obf[c[q - 1]];

    /*
       (a) the m - q unmodelled criteria set at their
       reference levels tau_i (i.e. no perceived gains or
       losses)
    */

    for (int i = q; i < m; i++) {
        zhat[i] = ip[c[i]];
    }

    return zhat;
}
double machineDM::stewart_value_function(const vector_double &obj, const vector_double &tau) const
{
    double sum = 0.0;
    double st;
    vector_double w = this->pref.weights;
    for (int i = 0; i < obj.size(); i++) {
        if (obj[i] <= tau[i]) {
            st = lambda[i] * (exp(alpha[i] * obj[i]) - 1.0) / (exp(alpha[i] * tau[i]) - 1.0);
        } else {
            st = lambda[i]
                 + (1.0 - lambda[i]) * (1.0 - exp(-beta[i] * (obj[i] - tau[i])))
                       / (1.0 - exp(-beta[i] * (1.0 - tau[i])));
        }
        sum += w[i] * st;
    }
    return sum;
}

// M:It's been supposed that the training data is a vector of decision vectors. and their last member is the
// dm_evaluated value

double machineDM::dm_evaluate(
    vector_double &obj) // moved onst vector_double &alpha, const vector_double &beta, const vector_double
                        // &lambda, double gamma, double sigma, double delta, int q to machineDM calss parameters
{
    int m = obj.size();
    vector_double tau = this->pref.ideal_point; // M: I'm not sure if this is a right way to access the ideal_point
    std::vector<int> c(m);

    if (q < m) {
        c = machineDM::select_criteria_subset();
    } else {
        assert(q == m);
        for (size_t i = 0; i < m; i++) {
            c[i] = i;
        }
    }

    vector_double z_mod = machineDM::modify_criteria(obj, c);

    /*
      (b) the addition of a noise term, normally
      distributed with zero mean and a variance of
      sigma^2 (which will be a specified model parameter),
    */
    double noise = machineDM::Rand_normal(0.0, sigma * sigma);

    /* (c) a shift in the reference levels tau_i from the ‘ideal’
       positions by an amount \delta, which may be
       positive or negative (and which is also a
       specified model parameter).
    */
    vector_double tau_mod(m);
    for (int i = 0; i < m; i++) {
        tau_mod[i] = tau[i] + delta;
    }

    double estim_v = noise + machineDM::stewart_value_function(z_mod, tau_mod);
    return estim_v;
}
double machineDM::Rand_normal(double mean, double sd)

/** REFERENCES (http://www.cast.uark.edu/~kkvamme/ACN37.htm)

 Hays, W.L.
 1988 Statistics (4th ed.). Holt, Rinehart and Winston, New York.

 Hodder, Ian (ed.)
 1978 Simulation Studies in Archaeology. Cambridge University Press, Cambridge.

 Olkin, I., L.J. Gleser, and C. Derman
 1980 Probability Models and Applications. Macmillan Publishing, New York.

 Ross, Sheldon M.
 1989 Introduction to Probability Models (4th ed.). Academic Press, Boston.
**/

{
    int i;
    double x;

    for (x = 0.0, i = 0; i < 12; i++) {
        x += std::rand() / RAND_MAX;
    }
    return (mean + sd * (x - 6.0));
}

vector_double machineDM::fitness(const vector_double &solution) const
{
    // FIXME: Apply biases
    vector_double f = prob.fitness(solution);

    return f;
}

double machineDM::value(const vector_double &solution)
{
    vector_double f = fitness(solution);
    // FIXME: Apply biases
    if (this->mode == NI) {
        return pref.value(f);
    }
    return dm_evaluate(f);
}

double machineDM::true_value(const vector_double &solution) const
{
    vector_double f = prob.fitness(solution);
    return pref.value(f);
}

std::vector<size_t> machineDM::rank(const pagmo::population &pop) const
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
