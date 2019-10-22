#ifndef PAGMO_ALGORITHMS_LEARNER_HPP
#define PAGMO_ALGORITHMS_LEARNER_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithms/machineDM.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>

extern "C" {
#include "pagmo/utils/svm_common.h"
#include "pagmo/utils/svm_learn.h"
}

namespace pagmo
{
class PAGMO_DLL_PUBLIC svm
{
public:
    svm(machineDM &dm, int start = 0, int cv_k = 0); // svm(machineDM &dm, int start, int argc, char **argv)
    //     : start(start), mdm(dm)   // M: WE may also define svm as a derived class od machineDM so it can access the
    //                               // utility funcionts and etc
    // {
    //     // parse_command_line(start, argc, argv, &verbosity, &learn_parm, &kernel_parm);
    //
    //     init();
    // };

    ~svm();

    // Parameters of SVMlight
    LEARN_PARM learn_parm;
    KERNEL_PARM kernel_parm;

    MODEL *m_model;
    DOC **m_examples;
    long m_num_examples;
    double *m_targets;

    DOC ***m_train_examples;
    double **m_train_targets;
    long *m_num_train_examples;

    DOC ***m_test_examples;
    double **m_test_targets;
    long *m_num_test_examples;

    long m_max_feature_id; // Parameters of SVMlight

    // k for cross validation model selection (0 = loo)
    int m_cv_k;
    vector_double m_pref;
    // whether to perform model selection
    bool m_do_model_selection;
    double m_results_threshold;
    machineDM mdm;
    int m_curr_iteration;
    // void setPreferences(population &pop, int start, int popsize, int objsize, bool rankerprefs);
    double preference(vector_double &obj, int objsize); //, bool rankerprefs
    void print_help();
    double train(pagmo::population &pop, int start, int popsize, int objsize);
    int start;
    // protected:

    void free_examples(DOC **examples, long num_examples); // Deletes the training examples
    // void init();
    void setRankingPreferences(population &pop, int start, int popsize, int objsize);

    double do_model_selection();
    void do_model_evaluation(double *best_performance, int *best_kernel_type, int *best_degree, double *best_gamma);
    double do_cross_validation();
    void updateSvmProblem(DOC ***examples_p, double **targets_p, long *num_examples_p, long qid, pagmo::population &pop,
                          int popstart, int popsize, int objsize);
    DOC *create_instance(int instnum, vector_double &obj, int objsize, int qid);
    void updateCVProblems(population &pop, int popstart, int popsize, int objsize);
    void train(DOC **examples, double *targets, long num_examples);
    double test(DOC **examples, double *targets, long num_examples);
    void parse_command_line(int start, int argc, char *argv[], long *verbosity, LEARN_PARM *learn_parm,
                            KERNEL_PARM *kernel_parm);
    void copy_learn_parm(LEARN_PARM *src_learn_parm, LEARN_PARM *dst_learn_parm);
    void write_examples(std::ostream &out, DOC **examples, double *targets, long num_examples);
    std::vector<int> sort(vector_double &x);
    double corrcoeff(const std::vector<int> &x1, const std::vector<int> &x2);
    void wait_any_key();
};

} // namespace pagmo

#endif
