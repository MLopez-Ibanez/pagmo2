#ifndef PAGMO_ALGORITHMS_MACHINEDM_HPP
#define PAGMO_ALGORITHMS_MACHINEDM_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>

#define NI 1  // No interaction
#define ITV 2 // Interaction with true value
#define IMV 3 // Interaction with modified value

namespace pagmo
{
class PAGMO_DLL_PUBLIC value_function
{
public:
    virtual double value(const std::vector<double> &) const = 0;
    std::vector<double> weights;
    std::vector<double> ideal_point;

protected:
    value_function(std::vector<double> w, std::vector<double> ip = {0, 0}) : weights(w), ideal_point(ip){};
};
// moved tthe steart as a function to machineDM
// class PAGMO_DLL_PUBLIC stewart_value_function : public value_function
// {
// public:
//     stewart_value_function(std::vector<double> w) : value_function(w){};
//     double value(const vector_double &obj, const vector_double &alpha, const vector_double &beta,
//                  const vector_double &lambda, const vector_double &tau) const;
// };

class PAGMO_DLL_PUBLIC linear_value_function : public value_function
{
public:
    linear_value_function(std::vector<double> w) : value_function(w){};
    double value(const std::vector<double> &) const;
};

class PAGMO_DLL_PUBLIC poly_value_function
    : public value_function // Mahdi: The value function is protected, why we have public here?
{
private:
    std::vector<double> weights;
    std::vector<std::vector<int>> degrees;

public:
    poly_value_function(std::vector<double> w) : value_function(w)
    {
        static const size_t npos = -1;
        std::string str;
        int obfs;
        std::cout << " Enter a polynomial without like terms\n";
        std::cout << "(use the letter x. for ex.: -x1^4+x2X3^4)\n";
        std::cout << "\nEnter: ";
        std::cin >> str;
        if (str == "") {
            std::cout << "No function is defined\n";
        }
        size_t strSize = str.size();
        std::cout << "Enter total number of objectives/criteria\n";
        std::cout << "\nEnter: ";
        std::cin >> obfs;

        //	How many monomials has the polynomial?
        int k = 1;
        size_t i, j;
        for (i = 1; i < strSize; ++i)
            if (str[i] == '+' || str[i] == '-') k++;
        size_t monoms = k;

        //	Signs "+" are necessary for the string parsing
        if (isdigit(str[0])) str.insert(0, "+");
        if (str[0] == 'x') str.insert(0, "+");
        str.append("+");
        strSize = str.size();

        //	Extracting the monomials as monomStr
        k = 0;
        j = 0;
        std::string monomStr[monoms];
        for (i = 1; i < strSize; ++i)
            if (str[i] == '+' || str[i] == '-') {
                monomStr[k++] = str.substr(j, i - j);
                j = i;
            }

        //  Monomials' formatting i.e. to have all the same form: coefficientX^exponent
        for (i = 0; i < monoms; ++i) {
            if (monomStr[i][1] == 'x')      // x is after the +/- sign
                monomStr[i].insert(1, "1"); //& gets 1 as coefficient
            bool flag = false;              // assuming that x is not present
            size_t len = monomStr[i].size();
            for (j = 1; j < len; ++j)
                if (monomStr[i][j] == 'x') // but we test this
                {
                    flag = true;                  //& if x is present
                    if (j == len - 1)             //& is the last
                        monomStr[i].append("^1"); // it gets exponent 1
                    break;                        //& exit from j_loop
                }
            if (!flag)                     // if x is not present: we have a constant term
                monomStr[i].append("x^0"); // who gets "formatting"
        }

        // extracting weights and degrees from monomStr
        weights.resize(monoms, 1); // we have monoms weights

        // extraction of weights
        for (i = 0; i < monoms; i++) {
            if (monomStr[i].find('x')
                == std::string::npos) { // because of some errors I had to replace string::npos with -1;
                weights[i] = stoi(monomStr[i]);
                continue;
            }
            for (j = 0; j < monomStr[i].size(); i++) {
                if (monomStr[i][j] == 'x' || monomStr[i][j] == 'X') {
                    if (j == 1) {
                        break;
                    } else {
                        weights[i] = stoi(monomStr[i].substr(0, j - 1));
                        break;
                    }
                }
            }
        }

        // extractiion of degress
        // degrees has monoms elemets (for each monomial of the polonomial), each element is a vector of m elements
        degrees.resize(monoms, std::vector<int>(obfs)); // where m is the number of obfs each of the m elements specify
                                                        // the degree of that obf in this monomial
        for (int i = 0; i < monoms; i++) {
            j = 0;
            int xindex1, xindex2, pindex, o;
            while (j < monomStr[i].size()) {
                if (monomStr[i].find('x') == std::string::npos) {
                    break;
                }
                xindex2 = monomStr[i].find('x', xindex1 + 1);
                pindex = monomStr[i].find('^', xindex1 + 1);
                o = stoi(monomStr[i].substr(xindex1 + 1, std::min(xindex2, pindex)));
                degrees[i][o] = 1;
                j = std::min(xindex2, pindex);
                if (xindex2 > pindex) {
                    degrees[i][o] = stoi(monomStr[i].substr(pindex + 1, xindex2));
                    j = xindex2;
                }
            }
        }
    };
    double value(const std::vector<double> &) const;
};

class PAGMO_DLL_PUBLIC quadratic_value_function : public value_function
{
public:
    quadratic_value_function(std::vector<double> w, std::vector<double> ip)
        : value_function(w, ip) // maybe we can delete the ip(idealpoint) from the input, get it in the constructor by
                                // pgamo::ideal(get_f)
          {};
    double value(const std::vector<double> &) const;
};

class PAGMO_DLL_PUBLIC tchebycheff_value_function : public value_function
{
public:
    tchebycheff_value_function(std::vector<double> w, std::vector<double> ip)
        : value_function(w, ip) // maybe we can delete the ip(idealpoint) from the input, get it in the constructor by
                                // pgamo::ideal(get_f)
          {};
    double value(const std::vector<double> &) const;
};

/// Machine Decision Maker
/**
 * FIXME: DOCUMENT ME !
 */
class PAGMO_DLL_PUBLIC machineDM // M: we can bring all the parameters such as alpha, beta , sigam and etc here in the
                                 // parameters of machineDM
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

    machineDM(problem &prob, value_function &pref, unsigned mode, vector_double alpha, vector_double beta,
              vector_double lambda, vector_double tau, double gamma, double sigma, double delta, int q,
              unsigned seed = pagmo::random_device::next())
        : prob(prob), pref(pref), mode(mode), alpha(alpha), beta(beta), lambda(lambda), tau(tau), gamma(gamma),
          sigma(sigma), delta(delta), q(q), rand_normal(0., sigma * sigma) // to generate a normally distributed number
    {
        m_e.seed(seed);
        m_seed = seed;
    };

    /**
     * Evaluate fitness (objective vector) according to decision maker.
     *
     *
     **/

    std::vector<vector_double> trainFile;
    vector_double fitness(vector_double);

    vector_double get_weights() const;
    /* Select a subset of q criteria from m true criteria with probability
   proportional to their weights.  */
    std::vector<int> select_criteria_subset() const;
    int roulette_wheel(const vector_double &w) const;
    vector_double modify_criteria(const vector_double &obf, const std::vector<int> &c);

    double dm_evaluate(const vector_double &obj);
    double stewart_value_function(const vector_double &obj, const vector_double &tau) const;
    double Rand_normal(double mean, double sd);

    /**
     * Solution value according to DM.
     *
     *
     **/
    double value(vector_double);

    /**
     * True solution value according to DM's preference function.
     *
     *
     **/
    double true_value(const vector_double &) const;

    std::vector<size_t> rank(const pagmo::population &pop) const;
    template <typename Archive>
    void serialize(Archive &ar, unsigned);
    vector_double alpha, beta, lambda, tau;
    double gamma, sigma, delta;
    int q;
    problem &prob;
    value_function &pref;
    unsigned mode;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    std::normal_distribution<double> rand_normal;
}; // MachineDM

} // namespace pagmo

// FIXME: Not working
// PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::machineDM)

#endif
