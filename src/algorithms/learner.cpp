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
#include <pagmo/algorithms/learner.hpp>
//#include <pagmo/algorithms/matrix.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

#ifdef DEBUG
#include <sstream>
#endif

namespace pagmo
{
svm::svm(machineDM &dm, int start) // svm(machineDM &dm, int start, int argc, char **argv)
    : start(start), mdm(dm)        // M: WE may also define svm as a derived class od machineDM so it can access the
                                   // utility funcionts and etc
{
    // parse_command_line(start, argc, argv, &verbosity, &learn_parm, &kernel_parm);

    init();
}
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
void svm::free_examples(DOC **examples, long num_examples)
{
    if (examples) {
        for (long i = 0; i < num_examples; i++)
            free_example(examples[i], 1);
        free(examples);
    }
}

void svm::init()
{

    m_num_examples = 0;
    m_max_feature_id = 0;
    m_model = NULL;
    m_examples = NULL;
    m_targets = NULL;
    m_train_examples = (DOC ***)calloc(m_cv_k, sizeof(DOC **));
    m_test_examples = (DOC ***)calloc(m_cv_k, sizeof(DOC **));
    m_train_targets = (double **)calloc(m_cv_k, sizeof(double *));
    m_test_targets = (double **)calloc(m_cv_k, sizeof(double *));
    m_num_train_examples = (long *)calloc(m_cv_k, sizeof(long));
    m_num_test_examples = (long *)calloc(m_cv_k, sizeof(long));
}

// Sets the preference value or comparison values (somehow Non domination value based on preference values)
// void svm::setPreferences(population &pop, int start, int popsize, int objsize,
//                          bool rankerprefs) // M: where is this function used? not useful
// {
//     if (!rankerprefs) // M: if randerprefs is false then setRankingPrefereces? shouldn't it be vice versa?
//         setRankingPreferences(pop, start, popsize, objsize);
//     else
//         for (int i = start; i < popsize; i++) {
//             // compute preference for current individual
//             pop.m_pref[i] = mdm.value(pop.m_f[i]);
//         };
// }

// comparing the solutions by DM 2 by 2
void svm::setRankingPreferences(population &pop, int start, int popsize, int objsize)
{
    // init ranking preferences to 0
    for (int i = start; i < popsize; i++) {
        // init preference for current individual
        pop.m_pref[i] = 0;
    }
    double pref;
    std::vector<vector_double> f = pop.get_f();
    for (int i = start; i < popsize; i++)
        for (int j = i + 1; j < popsize; j++) {
            pref = mdm.value(f[i]) - mdm.value(f[j]); // M: instead of userPreference in old files

            if (pref <= 0) pop.m_pref[i]--;
            if (pref >= 0) pop.m_pref[j]--;
        }
}

double svm::train(pagmo::population &pop, int start, int popsize, int objsize)
{
    double results = 0.;

    m_max_feature_id = objsize;
    // add current individuals to training set
    updateSvmProblem(&m_examples, &m_targets, &m_num_examples, m_curr_iteration, pop, start, popsize, objsize);

    if (m_cv_k)
        // add current individuals to folds
        updateCVProblems(pop, start, popsize, objsize);

#ifdef DEBUG
    // printing datasets
    ofstream out("train");
    write_examples(out, m_examples, m_targets, m_num_examples);
    out.close();
    out.clear();
    for (int k = 0; k < m_cv_k; k++) {
        ostringstream ostr;
        ostr << "train" << k;
        out.open(ostr.str().c_str());
        write_examples(out, m_train_examples[k], m_train_targets[k], m_num_train_examples[k]);
        out.close();
        out.clear();
        ostringstream oste;
        oste << "test" << k;
        out.open(oste.str().c_str());
        write_examples(out, m_test_examples[k], m_test_targets[k], m_num_test_examples[k]);
        out.close();
        out.clear();
    }
#endif

    std::cout << "\n\nStarting ranker training\n" << std::endl;

    if (m_do_model_selection) {
        // double c = learn_parm.svm_c;
        // learn_parm.svm_c = 1;
        results = do_model_selection();

        std::cout << "choosing: kernel=" << kernel_parm.kernel_type << " degree=" << kernel_parm.poly_degree
                  << " gamma=" << kernel_parm.rbf_gamma << std::endl;
        // m_do_model_selection = false;
        // learn_parm.svm_c = c;
    } else if (m_cv_k)
        // evaluate current parameter setting
        results = do_cross_validation();

    // train model
    train(m_examples, m_targets, m_num_examples);

    std::cout << "\n\nFinished ranker training\n" << std::endl;

    // DEBUG
    // write_model("debug.model",m_model);

    return results;
}

double svm::do_model_selection()
{
    double best_performance = -std::numeric_limits<double>::max();
    int best_kernel_type = kernel_parm.kernel_type;
    double best_gamma = kernel_parm.rbf_gamma;
    int best_degree = kernel_parm.poly_degree;

    // int maxdegree = 4;
    int maxdegree = 2;
    int numgammas = 7;
    double gammas[] = {1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3};

    // linear
    kernel_parm.kernel_type = 0;
    do_model_evaluation(&best_performance, &best_kernel_type, &best_degree, &best_gamma);
    if (best_performance >= m_results_threshold) return best_performance;

    // polynomial
    kernel_parm.kernel_type = 1;
    for (int d = 2; d <= maxdegree; d++) {
        kernel_parm.poly_degree = d;
        do_model_evaluation(&best_performance, &best_kernel_type, &best_degree, &best_gamma);
        if (best_performance >= m_results_threshold) return best_performance;
    }

    // gaussian
    kernel_parm.kernel_type = 2;
    for (int g = 0; g < numgammas; g++) {
        kernel_parm.rbf_gamma = gammas[g];
        do_model_evaluation(&best_performance, &best_kernel_type, &best_degree, &best_gamma);
        if (best_performance >= m_results_threshold) return best_performance;
    }

    kernel_parm.kernel_type = best_kernel_type;
    kernel_parm.poly_degree = best_degree;
    kernel_parm.rbf_gamma = best_gamma;

    return best_performance;
}

void svm::do_model_evaluation(double *best_performance, int *best_kernel_type, int *best_degree, double *best_gamma)
{

    double curr_performance = do_cross_validation();

    std::cout << "kernel=" << kernel_parm.kernel_type << " degree=" << kernel_parm.poly_degree
              << " gamma=" << kernel_parm.rbf_gamma << " cv_performance=" << curr_performance << std::endl;

    if (curr_performance > *best_performance) {
        *best_performance = curr_performance;
        *best_kernel_type = kernel_parm.kernel_type;
        *best_degree = kernel_parm.poly_degree;
        *best_gamma = kernel_parm.rbf_gamma;
    }
}
double svm::do_cross_validation()
{
    double avgperf = 0.;

    for (int k = 0; k < m_cv_k; k++) {
        train(m_train_examples[k], m_train_targets[k], m_num_train_examples[k]);
        avgperf += test(m_test_examples[k], m_test_targets[k], m_num_test_examples[k]);
    }

    return (avgperf / m_cv_k);
}

void svm::updateSvmProblem(DOC ***examples_p, double **targets_p, long *num_examples_p, long qid,
                           pagmo::population &pop, int popstart, int popsize, int objsize)
{
    std::vector<vector_double> f = pop.get_f();
    // update set size
    long exstart = *num_examples_p;
    *num_examples_p += popsize - popstart;

    // allocate memory
    if (*examples_p)
        *examples_p = (DOC **)realloc(*examples_p, sizeof(DOC *) * (*num_examples_p));
    else
        *examples_p = (DOC **)calloc((*num_examples_p), sizeof(DOC *));

    // allocate memory
    if (*targets_p)
        *targets_p = (double *)realloc(*targets_p, sizeof(double) * (*num_examples_p));
    else
        *targets_p = (double *)calloc((*num_examples_p), sizeof(double));

    for (int i = popstart, e = exstart; i < popsize; i++, e++) {
        (*targets_p)[e] = pop.m_pref[i];
        (*examples_p)[e] = create_instance(e, f[i], objsize, qid + 1);
    }
}

DOC *svm::create_instance(int instnum, vector_double &obj, int objsize, int qid)
{
    WORD *words = (WORD *)my_malloc(sizeof(WORD) * (objsize + 1));

    DOC *doc;

    for (int i = 0; i < objsize; i++) {
        words[i].wnum = i + 1;
        words[i].weight = obj[i];
    }
    words[objsize].wnum = 0;

    // create doc
    doc = create_example(instnum, qid, 0, 1., create_svector(words, (char *)"", 1.0));
    free(words);

    return doc;
}

void svm::updateCVProblems(population &pop, int popstart, int popsize, int objsize)
{
    if (popstart != 0) {
        std::cerr << "ERROR: assuming zero popstart" << std::endl;
        exit(0);
    }

    int foldsize = (popsize - popstart) / m_cv_k;

    // update train folds
    for (int k = 0; k < m_cv_k; k++) {
        for (int h = 0; h < m_cv_k; h++)
            if (h != k)
                updateSvmProblem(&m_train_examples[k], &m_train_targets[k], &m_num_train_examples[k], m_curr_iteration,
                                 pop, h * foldsize, h * foldsize + foldsize, objsize);
        if (m_curr_iteration % 2) {
            for (int h = m_cv_k - 1, pos = m_cv_k * foldsize; h >= 0 && pos < popsize; h--, pos++)
                if (h != k)
                    updateSvmProblem(&m_train_examples[k], &m_train_targets[k], &m_num_train_examples[k],
                                     m_curr_iteration, pop, pos, pos + 1, objsize);
        } else {
            for (int h = 0, pos = m_cv_k * foldsize; h < m_cv_k && pos < popsize; h++, pos++)
                if (h != k)
                    updateSvmProblem(&m_train_examples[k], &m_train_targets[k], &m_num_train_examples[k],
                                     m_curr_iteration, pop, pos, pos + 1, objsize);
        }
    }

    // update test folds
    for (int k = 0; k < m_cv_k; k++)
        updateSvmProblem(&m_test_examples[k], &m_test_targets[k], &m_num_test_examples[k], m_curr_iteration, pop,
                         k * foldsize, k * foldsize + foldsize, objsize);
    if (m_curr_iteration % 2) {
        for (int k = m_cv_k - 1, pos = m_cv_k * foldsize; k >= 0 && pos < popsize; k--, pos++)
            updateSvmProblem(&m_test_examples[k], &m_test_targets[k], &m_num_test_examples[k], m_curr_iteration, pop,
                             pos, pos + 1, objsize);
    } else {
        for (int k = 0, pos = m_cv_k * foldsize; k < m_cv_k && pos < popsize; k++, pos++)
            updateSvmProblem(&m_test_examples[k], &m_test_targets[k], &m_num_test_examples[k], m_curr_iteration, pop,
                             pos, pos + 1, objsize);
    }
}
double svm::preference(vector_double &obj, int objsize) // deleted bool rankerprefs from the function
{
    double pref;
    DOC *x;

    // if (!m_use_gold_preference && rankerprefs) {

    x = create_instance(0, obj, objsize, 1);

    if (m_model->kernel_parm.kernel_type == 0) /* linear kernel */
        pref = classify_example_linear(m_model, x);
    else
        pref = classify_example(m_model, x);

    free_example(x, 1);
    // } else {
    //     pref = this->dm.value(obj);
    // }
    return pref;
}

void svm::train(DOC **examples, double *targets, long num_examples)
{
    KERNEL_CACHE *kernel_cache;

    LEARN_PARM curr_learn_parm;

    copy_learn_parm(&learn_parm, &curr_learn_parm);

    if (m_model) free_model(m_model, 0);
    m_model = (MODEL *)my_malloc(sizeof(MODEL));

    if (kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
        kernel_cache = NULL;
    } else {
        /* Always get a new kernel cache. It is not possible to use the
           same cache for two different training runs */
        kernel_cache = kernel_cache_init(num_examples, curr_learn_parm.kernel_cache_size);
    }

    svm_learn_ranking(examples, targets, num_examples, m_max_feature_id, &curr_learn_parm, &kernel_parm, &kernel_cache,
                      m_model);

    if (kernel_cache) {
        /* Free the memory used for the cache. */
        kernel_cache_cleanup(kernel_cache);
    }

    if (m_model->kernel_parm.kernel_type == 0) { /* linear kernel */
        /* compute weight vector */
        add_weight_vector_to_linear_model(m_model);

        std::cout << "learned polynomial:" << std::endl;
        // NOTE: lin_weights contains totwords+1 elements.
        // First element could be bias term b ?
        for (int i = 1; i <= m_model->totwords; i++) {
            if (i > 1) std::cout << " + ";
            std::cout << m_model->lin_weights[i];
            std::cout << " * x" << i - 1;
        }
        std::cout << std::endl;
    }
}

double svm::test(DOC **examples, double *targets, long num_examples)
{
    double l;

    if (num_examples <= 1) return 1.;

    vector_double preds(num_examples);
    vector_double positions(num_examples);
    for (long i = 0; i < num_examples; i++) {

        if (m_model->kernel_parm.kernel_type == 0) /* linear kernel */
            l = classify_example_linear(m_model, examples[i]);
        else
            l = classify_example(m_model, examples[i]);

        preds[i] = l;
        positions[i] = targets[i];
    }

    std::vector<int> preds_sort = sort(preds);
    std::vector<int> positions_sort = sort(positions);

    return corrcoeff(
        preds_sort,
        positions_sort); // M: I believe this is the correlation coefficient. thus I wrote a function to calculate this.
}

// sort function get a vector, sorts it in an increasing order and returns the indexes of the original vector in the
// incrasing order
std::vector<int>
svm::sort(vector_double
              &x) // M: I need to move these kinds of functions to a utilitiy header. like reoulette wheel selection.

{
    int s = x.size();
    std::vector<int> order(s);
    for (int r = 0; r < s; r++)
        order[r] = r;

    for (int pass = 0; pass < s; pass++)
        for (int r = 0; r < s; r++)
            if (x[r] < x[r + 1]) {
                double hold = x[r];
                x[r] = x[r + 1];
                x[r + 1] = hold;
                int pos = order[r];
                order[r] = order[r + 1];
                order[r + 1] = pos;
            }
    return order;
}

double svm::corrcoeff(const std::vector<int> &X, const std::vector<int> &Y)
{
    int sum_X = 0, sum_Y = 0, sum_XY = 0;
    int squareSum_X = 0, squareSum_Y = 0;
    int n = X.size();
    for (int i = 0; i < n; i++) {
        // sum of elements of array X.
        sum_X = sum_X + X[i];

        // sum of elements of array Y.
        sum_Y = sum_Y + Y[i];

        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];

        // sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i];
        squareSum_Y = squareSum_Y + Y[i] * Y[i];
    }

    // use formula for calculating correlation coefficient.
    double corr = (double)(n * sum_XY - sum_X * sum_Y)
                  / sqrt((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y));

    return corr;
}
void wait_any_key()
{
    printf("\n(more)\n");
    (void)getc(stdin);
}
void svm::parse_command_line(int start, int argc, char *argv[], long *verbosity, LEARN_PARM *learn_parm,
                             KERNEL_PARM *kernel_parm)
{
    long i;
    char type[100];

    /* set default */
    strcpy(learn_parm->predfile, "trans_predictions");
    strcpy(learn_parm->alphafile, "");
    (*verbosity) = 1;
    learn_parm->biased_hyperplane = 1;
    learn_parm->sharedslack = 0;
    learn_parm->remove_inconsistent = 0;
    learn_parm->skip_final_opt_check = 0;
    learn_parm->svm_maxqpsize = 10;
    learn_parm->svm_newvarsinqp = 0;
    learn_parm->svm_iter_to_shrink = -9999;
    learn_parm->maxiter = 100000;
    learn_parm->kernel_cache_size = 40;
    learn_parm->svm_c = 0.0;
    learn_parm->eps = 0.1;
    learn_parm->transduction_posratio = -1.0;
    learn_parm->svm_costratio = 1.0;
    learn_parm->svm_costratio_unlab = 1.0;
    learn_parm->svm_unlabbound = 1E-5;
    learn_parm->epsilon_crit = 0.001;
    learn_parm->epsilon_a = 1E-15;
    learn_parm->compute_loo = 0;
    learn_parm->rho = 1.0;
    learn_parm->xa_depth = 0;
    kernel_parm->kernel_type = 0;
    kernel_parm->poly_degree = 3;
    kernel_parm->rbf_gamma = 1.0;
    kernel_parm->coef_lin = 1;
    kernel_parm->coef_const = 1;
    strcpy(kernel_parm->custom, "empty");
    strcpy(type, "p");

    m_do_model_selection = true;
    m_cv_k = 3;

    for (i = start; (i < argc) && ((argv[i])[0] == '-'); i++) {
        switch ((argv[i])[1]) {
            case '?':
                print_help();
                exit(0);
            case 'z':
                i++;
                strcpy(type, argv[i]);
                break;
            case 'v':
                i++;
                (*verbosity) = atol(argv[i]);
                break;
            case 'b':
                i++;
                learn_parm->biased_hyperplane = atol(argv[i]);
                break;
            case 'i':
                i++;
                learn_parm->remove_inconsistent = atol(argv[i]);
                break;
            case 'f':
                i++;
                learn_parm->skip_final_opt_check = !atol(argv[i]);
                break;
            case 'q':
                i++;
                learn_parm->svm_maxqpsize = atol(argv[i]);
                break;
            case 'n':
                i++;
                learn_parm->svm_newvarsinqp = atol(argv[i]);
                break;
            case '#':
                i++;
                learn_parm->maxiter = atol(argv[i]);
                break;
            case 'h':
                i++;
                learn_parm->svm_iter_to_shrink = atol(argv[i]);
                break;
            case 'm':
                i++;
                learn_parm->kernel_cache_size = atol(argv[i]);
                break;
            case 'c':
                i++;
                learn_parm->svm_c = atof(argv[i]);
                break;
            case 'w':
                i++;
                learn_parm->eps = atof(argv[i]);
                break;
            case 'p':
                i++;
                learn_parm->transduction_posratio = atof(argv[i]);
                break;
            case 'j':
                i++;
                learn_parm->svm_costratio = atof(argv[i]);
                break;
            case 'e':
                i++;
                learn_parm->epsilon_crit = atof(argv[i]);
                break;
            case 'o':
                i++;
                learn_parm->rho = atof(argv[i]);
                break;
            case 'k':
                i++;
                learn_parm->xa_depth = atol(argv[i]);
                break;
            case 'x':
                i++;
                learn_parm->compute_loo = atol(argv[i]);
                break;
            case 't':
                i++;
                kernel_parm->kernel_type = atol(argv[i]);
                break;
            case 'd':
                i++;
                kernel_parm->poly_degree = atol(argv[i]);
                break;
            case 'g':
                i++;
                kernel_parm->rbf_gamma = atof(argv[i]);
                break;
            case 's':
                i++;
                kernel_parm->coef_lin = atof(argv[i]);
                break;
            case 'r':
                i++;
                kernel_parm->coef_const = atof(argv[i]);
                break;
            case 'u':
                i++;
                strcpy(kernel_parm->custom, argv[i]);
                break;
            case 'l':
                i++;
                strcpy(learn_parm->predfile, argv[i]);
                break;
            case 'a':
                i++;
                strcpy(learn_parm->alphafile, argv[i]);
                break;
            case 'M':
                m_do_model_selection = true;
                break;
            case 'V':
                i++;
                m_cv_k = atoi(argv[i]);
                break;
            default:
                printf("\nUnrecognized option %s!\n\n", argv[i]);
                print_help();
                exit(0);
        }
    }

    if (m_do_model_selection && !m_cv_k) {
        printf("\nCannot perform model selection with 0 folds.\n");
        wait_any_key();
        print_help();
        exit(0);
    }

    if (learn_parm->svm_iter_to_shrink == -9999) {
        if (kernel_parm->kernel_type == LINEAR)
            learn_parm->svm_iter_to_shrink = 2;
        else
            learn_parm->svm_iter_to_shrink = 100;
    }
    if (strcmp(type, "c") == 0) {
        learn_parm->type = CLASSIFICATION;
    } else if (strcmp(type, "r") == 0) {
        learn_parm->type = REGRESSION;
    } else if (strcmp(type, "p") == 0) {
        learn_parm->type = RANKING;
    } else if (strcmp(type, "o") == 0) {
        learn_parm->type = OPTIMIZATION;
    } else if (strcmp(type, "s") == 0) {
        learn_parm->type = OPTIMIZATION;
        learn_parm->sharedslack = 1;
    } else {
        printf("\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference "
               "ranking.\n",
               type);
        wait_any_key();
        print_help();
        exit(0);
    }
    if ((learn_parm->skip_final_opt_check) && (kernel_parm->kernel_type == LINEAR)) {
        printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
        learn_parm->skip_final_opt_check = 0;
    }
    if ((learn_parm->skip_final_opt_check) && (learn_parm->remove_inconsistent)) {
        printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if ((learn_parm->svm_maxqpsize < 2)) {
        printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n", learn_parm->svm_maxqpsize);
        wait_any_key();
        print_help();
        exit(0);
    }
    if ((learn_parm->svm_maxqpsize < learn_parm->svm_newvarsinqp)) {
        printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n", learn_parm->svm_maxqpsize);
        printf("new variables [%ld] entering the working set in each iteration.\n", learn_parm->svm_newvarsinqp);
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->svm_iter_to_shrink < 1) {
        printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",
               learn_parm->svm_iter_to_shrink);
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->svm_c < 0) {
        printf("\nThe C parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->transduction_posratio > 1) {
        printf("\nThe fraction of unlabeled examples to classify as positives must\n");
        printf("be less than 1.0 !!!\n\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->svm_costratio <= 0) {
        printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->epsilon_crit <= 0) {
        printf("\nThe epsilon parameter must be greater than zero!\n\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if (learn_parm->rho < 0) {
        printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
        printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
        printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
        wait_any_key();
        print_help();
        exit(0);
    }
    if ((learn_parm->xa_depth < 0) || (learn_parm->xa_depth > 100)) {
        printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
        printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
        printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
        wait_any_key();
        print_help();
        exit(0);
    }
}

void svm::write_examples(std::ostream &out, DOC **examples, double *targets, long num_examples)
{
    for (long i = 0; i < num_examples; i++) {
        out << targets[i] << " qid:" << examples[i]->queryid;
        for (SVECTOR *v = examples[i]->fvec; v; v = v->next) {
            for (long j = 0; (v->words[j]).wnum; j++) {
                out << " " << (v->words[j]).wnum << ":" << (v->words[j]).weight;
            }
            if (v->userdefined)
                out << " #" << v->userdefined << std::endl;
            else
                out << " #" << std::endl;
        }
    }
}

void copy_learn_parm(LEARN_PARM *src_learn_parm, LEARN_PARM *dst_learn_parm)
{
    strcpy(dst_learn_parm->predfile, src_learn_parm->predfile);
    strcpy(dst_learn_parm->alphafile, src_learn_parm->predfile);

    dst_learn_parm->type = src_learn_parm->type;
    dst_learn_parm->biased_hyperplane = src_learn_parm->biased_hyperplane;
    dst_learn_parm->sharedslack = src_learn_parm->sharedslack;
    dst_learn_parm->remove_inconsistent = src_learn_parm->remove_inconsistent;
    dst_learn_parm->skip_final_opt_check = src_learn_parm->skip_final_opt_check;
    dst_learn_parm->svm_maxqpsize = src_learn_parm->svm_maxqpsize;
    dst_learn_parm->svm_newvarsinqp = src_learn_parm->svm_newvarsinqp;
    dst_learn_parm->svm_iter_to_shrink = src_learn_parm->svm_iter_to_shrink;
    dst_learn_parm->maxiter = src_learn_parm->maxiter;
    dst_learn_parm->kernel_cache_size = src_learn_parm->kernel_cache_size;
    dst_learn_parm->svm_c = src_learn_parm->svm_c;
    dst_learn_parm->eps = src_learn_parm->eps;
    dst_learn_parm->transduction_posratio = src_learn_parm->transduction_posratio;
    dst_learn_parm->svm_costratio = src_learn_parm->svm_costratio;
    dst_learn_parm->svm_costratio_unlab = src_learn_parm->svm_costratio_unlab;
    dst_learn_parm->svm_unlabbound = src_learn_parm->svm_unlabbound;
    dst_learn_parm->epsilon_crit = src_learn_parm->epsilon_crit;
    dst_learn_parm->epsilon_a = src_learn_parm->epsilon_a;
    dst_learn_parm->compute_loo = src_learn_parm->compute_loo;
    dst_learn_parm->rho = src_learn_parm->rho;
    dst_learn_parm->xa_depth = src_learn_parm->xa_depth;
}

} // namespace pagmo
