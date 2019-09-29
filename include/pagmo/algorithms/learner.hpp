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
#include "svm_rank/svm_light/svm_common.h"
#include "svm_rank/svm_light/svm_learn.h"
}

namespace pagmo
{
class PAGMO_DLL_PUBLIC svm
{
    svm::svm(machineDM dm, int start, int argc, char **argv)
        : start(start), mdm(dm) // M: WE may also define svm as a derived class od machineDM so it can access the
                                // utility funcionts and etc
    {
        parse_command_line(start, argc, argv, &verbosity, &learn_parm, &kernel_parm);

        init();
    };

    svm::~svm()
    {
        free_model(m_model, 0);
        free_examples(m_examples, m_num_examples);
        free(m_targets);

        for (int k = 0; k < m_cv_k; k++) {
            free_examples(m_train_examples[k], m_num_train_examples[k]);
            free_examples(m_test_examples[k], m_num_test_examples[k]);
            if (m_train_targets[k]) free(m_train_targets[k]);
            if (m_test_targets[k]) free(m_test_targets[k]);
        }

        free(m_train_examples);
        free(m_test_examples);
        free(m_train_targets);
        free(m_test_targets);
        free(m_num_train_examples);
        free(m_num_test_examples);
    }

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

    // whether to perform model selection
    bool m_do_model_selection;
    double m_results_threshold;

public:
    void setPreferences(population &pop, int start, int popsize, int objsize, bool rankerprefs);
    double preference(double *obj, int objsize, bool rankerprefs);
    static void print_help();

protected:
    void free_examples(DOC **examples, long num_examples); // Deletes the training examples
    void init();                                           // Iniializes the training set
    void setRankingPreferences(population *pop, int start, int popsize, int objsize);
    double train(population *pop, int start, int popsize, int objsize);
    double do_model_selection();
    void do_model_evaluation(double *best_performance, int *best_kernel_type, int *best_degree, double *best_gamma);
    double do_cross_validation();
    void updateSvmProblem(DOC ***examples_p, double **targets_p, long *num_examples_p, long qid, population *pop,
                          int popstart, int popsize, int objsize);
    DOC *create_instance(int instnum, double *obj, int objsize, int qid);
    void updateCVProblems(population *pop, int popstart, int popsize, int objsize);
    void train(DOC **examples, double *targets, long num_examples);
    double test(DOC **examples, double *targets, long num_examples);
    void parse_command_line(int start, int argc, char *argv[], long *verbosity, LEARN_PARM *learn_parm,
                            KERNEL_PARM *kernel_parm);
};

} // namespace pagmo

#endif
