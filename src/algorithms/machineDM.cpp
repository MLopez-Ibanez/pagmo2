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
#define IMV 3 // Interaction with modified value

namespace pagmo
{

double linear_value_function::value(const std::vector<double> &obj, std::vector<double>) const
{
    assert(this->weights.size() == obj.size());

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
double stewart_value_function::value(const std::vector<double> &obj, std::vector<double> tau)
    const // M: This tau is not hte same as the tau parameter of machineDM. should be reviewed
{
    double sum = 0.0;
    double st;
    vector_double w = this->weights;
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

double quadratic_value_function::value(const std::vector<double> &obj) const
{
    assert(this->weights.size() == obj.size());
    assert(this->ideal_point.size() == obj.size());

    double value = 0.;
    double w, ip, o;
    BOOST_FOREACH (boost::tie(w, ip, o), boost::combine(this->weights, this->ideal_point, obj)) {
        // FIXME: I think the weight should not be squared
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
        temp = std::max(w * std::abs(o - ip), value);
    }
    return value;
}
/* Select a subset of q criteria from m true criteria with probability
   proportional to their weights.  */
std::vector<int> machineDM::select_criteria_subset() const
{
    vector_double w = this->get_weights();
    std::vector<int> c(w.size());

    for (size_t i = 0; i < w.size(); i++) {
        int k = roulette_wheel(w);
        c[i] = k;
        w[k] = 0.0;
    }
    return c;
} // namespace pagmo
vector_double machineDM::get_weights() const
{
    return this->pref.weights;
}
// This roulette_wheel function returns an index of an element based on the
// probabilities constructed based on the values of the passed vector
int machineDM::roulette_wheel(const vector_double &w) const
{
    vector_double v(w.size());
    double sum = std::accumulate(w.begin(), w.end(), 0.0);
    double accumulated = 0.;
    for (int i = 0; i < v.size(); i++) {

        accumulated += w[i];
        v[i] = accumulated / sum;
    }
    // Random distributions
    double rndNumber = uniform_real_from_range(0, 1, m_e);
    for (int i = 0; i < v.size(); i++) {
        if (v[i] >= rndNumber) {
            return i;
        }
    }
}
vector_double machineDM::modify_criteria(
    const vector_double &obj,
    const std::vector<int> &c) // M: I have assumed that tau in previous vrsion is ideal point. thus I have changed it
{
    assert(gamma >= 0);
    assert(gamma < 1);
    int m = obj.size();
    vector_double ip = this->pref.ideal_point;
    vector_double zhat(m);
    for (int k = 1; k < q; k += 2) {
        assert(k < m);
        assert(c[k - 1] >= 0);
        assert(c[k - 1] < m);
        assert(c[k] >= 0);
        assert(c[k] < m);
        zhat[k - 1] = (1.0 - gamma) * obj[c[k - 1]] + gamma * obj[c[k]];
        zhat[k] = (1.0 - gamma) * obj[c[k]] + gamma * obj[c[k - 1]];
    }
    // if q is odd.
    if (q % 2) zhat[q - 1] = obj[c[q - 1]];

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

// M:It's been supposed that the training data is a vector of decision vectors. and their last member is the
// dm_evaluated value

double machineDM::dm_evaluate(
    const vector_double &obj1) // moved onst vector_double &alpha, const vector_double &beta, const vector_double
// &lambda, double gamma, double sigma, double delta, int q to machineDM calss parameters
{
    int m = obj1.size();
    std::vector<int> c(m);
    // const vector_double upperBound
    //     = this->prob.get_ub(); // M: Scaling the obj to [0,1] by deviding each obj value by its upperbound.
    vector_double obj;
    // std::pair<vector_double, vector_double> bounds = this->prob.get_bounds();

    if (q < m) {
        c = machineDM::select_criteria_subset();
    } else {
        assert(q == m);
        for (size_t i = 0; i < m; i++) {
            c[i] = i;
        }
    }

    const vector_double z_mod = machineDM::modify_criteria(obj1, c);
    // for (int i = 0; i < m; i++) {
    //     obj[i] = obj1[i] / upperBound[i];
    // }
    /* (c) a shift in the reference levels tau_i from the ‘ideal’
       positions by an amount \delta, which may be
       positive or negative (and which is also a
       specified model parameter).
    */
    const vector_double tau
        = this->pref.ideal_point; // M: I'm not sure if this is a right way to access the ideal_point
    vector_double tau_mod(m);
    for (int i = 0; i < m; i++) {
        tau_mod[i] = tau[i] + delta;
    }

    /*
  (b) the addition of a noise term, normally
  distributed with zero mean and a variance of
  sigma^2 (which will be a specified model parameter),
*/
    double noise = (sigma > 0) ? rand_normal(m_e) : 0.0;

    // FIXME: this should be the value function configured by the user.
    double estim_v = noise + this->st.value(z_mod, tau_mod);
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

vector_double machineDM::fitness(vector_double solution)
{
    assert(prob.get_nx() == solution.size());
    // FIXME: Apply biases
    vector_double f = prob.fitness(solution);

    return f;
}

double machineDM::value(vector_double obj)
{
    // vector_double f = fitness(solution);
    // FIXME: Apply biases
    if (this->mode == NI) {
        return pref.value(obj);
    }
    return dm_evaluate(obj);
}

double machineDM::true_value(const vector_double &solution) const
{
    assert(prob.get_nx() == solution.size());
    vector_double f = prob.fitness(solution);
    return pref.value(f);
}
void machineDM::setRankingPreferences(std::vector<vector_double> pop, std::vector<int> &m_pref, int start, int popsize,
                                      int objsize)
{
    m_pref.resize(popsize);
    // init ranking preferences to 0
    for (int i = start; i < popsize; i++) {
        // init preference for current individual
        m_pref[i] = 0;
    }
    double pref;
    // Scaling the population of objective values
    vector_double ideal = pagmo::ideal(pop);
    vector_double nadir = pagmo::nadir(pop);
    for (int i = start; i < popsize; i++)
        for (int j = i + 1; j < objsize; j++) {
            pop[i][j] = (pop[i][j] - ideal[j]) / (nadir[j] - ideal[j]);
        }
    // std::vector<vector_double> x = pop.get_x(); // M: I was assuming we are giving obj to MDM to evaluate. but right
    // now
    // we don't have such a function to evalute and compare objs
    for (int i = start; i < popsize; i++)
        for (int j = i + 1; j < popsize; j++) {
            pref = this->value(pop[i]) - this->value(pop[j]);

            if (pref <= 0) m_pref[i]--;
            if (pref >= 0) m_pref[j]--;
        }
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
