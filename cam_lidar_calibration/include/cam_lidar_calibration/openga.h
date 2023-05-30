// This library is free and distributed under
// Mozilla Public License Version 2.0.

#pragma once
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <ctime>
#include <string>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <limits>
#include <algorithm>
#include <functional>

#ifndef NS_EA_BEGIN
#define NS_EA_BEGIN                                                                                                    \
  namespace EA                                                                                                         \
  {
#define NS_EA_END }
#endif

NS_EA_BEGIN;

using std::cout;
using std::endl;
using std::function;
using std::runtime_error;
using std::vector;

enum class GA_MODE
{
  SOGA,
  IGA,
  NSGA_III
};

template <typename GeneType, typename MiddleCostType>
struct ChromosomeType
{
  GeneType genes;
  MiddleCostType middle_costs;  // individual costs
  double total_cost;            // for single objective
  vector<double> objectives;    // for multi-objective
};

template <typename GeneType, typename MiddleCostType>
struct GenerationType
{
  vector<ChromosomeType<GeneType, MiddleCostType>> chromosomes;
  double best_total_cost = (std::numeric_limits<double>::infinity());  // for single objective
  double average_cost = 0.0;                                           // for single objective

  int best_chromosome_index = -1;       // for single objective
  vector<int> sorted_indices;           // for single objective
  vector<vector<unsigned int>> fronts;  // for multi-objective
  vector<double> selection_chance_cumulative;
  double exe_time;
};

template <typename GeneType, typename MiddleCostType>
struct GenerationType_SO_abstract
{
  double best_total_cost = (std::numeric_limits<double>::infinity());  // for single objective
  double average_cost = 0.0;                                           // for single objective

  GenerationType_SO_abstract(const GenerationType<GeneType, MiddleCostType>& generation)
    : best_total_cost(generation.best_total_cost), average_cost(generation.average_cost)
  {
  }
};

class Matrix
{
  unsigned int n_rows, n_cols;
  vector<double> data;

public:
  Matrix() : n_rows(0), n_cols(0), data()
  {
  }

  Matrix(unsigned int n_rows, unsigned int n_cols) : n_rows(n_rows), n_cols(n_cols), data(n_rows * n_cols)
  {
  }

  void zeros()
  {
    std::fill(data.begin(), data.end(), 0);
  }

  void zeros(unsigned int rows, unsigned int cols)
  {
    n_rows = rows;
    n_cols = cols;
    data.assign(rows * cols, 0);
  }

  bool empty()
  {
    return (!n_rows) || (!n_cols);
  }

  unsigned int get_n_rows() const
  {
    return n_rows;
  }
  unsigned int get_n_cols() const
  {
    return n_cols;
  }

  void clear()
  {
    n_rows = 0;
    n_cols = 0;
    data.clear();
  }

  void set_col(unsigned int col_idx, const vector<double>& col_vector)
  {
    assert(col_vector.size() == n_rows && "Assigned column vector size mismatch.");
    for (unsigned int i = 0; i < n_rows; i++)
      (*this)(i, col_idx) = col_vector[i];
  }

  void set_row(unsigned int row_idx, const vector<double>& row_vector)
  {
    assert(row_vector.size() == n_cols && "Assigned row vector size mismatch.");
    for (unsigned int i = 0; i < n_cols; i++)
      (*this)(row_idx, i) = row_vector[i];
  }

  void get_col(unsigned int col_idx, vector<double>& col_vector) const
  {
    col_vector.resize(n_rows);
    for (unsigned int i = 0; i < n_rows; i++)
      col_vector[i] = (*this)(i, col_idx);
  }

  void get_row(unsigned int row_idx, vector<double>& row_vector) const
  {
    row_vector.resize(n_cols);
    for (unsigned int i = 0; i < n_cols; i++)
      row_vector[i] = (*this)(row_idx, i);
  }

  void operator=(const vector<vector<double>>& A)
  {
    unsigned int A_rows = (unsigned int)A.size();
    unsigned int A_cols = 0;
    if (A_rows > 0)
      A_cols = (unsigned int)A[0].size();
    n_rows = A_rows;
    n_cols = A_cols;
    if (n_rows > 0 && n_cols > 0)
    {
      data.resize(n_rows * n_cols);
      for (unsigned int i = 0; i < n_rows; i++)
      {
        assert(A[i].size() == A_cols && "Vector of vector does not have a constant row size! A21654616");
        for (unsigned int j = 0; j < n_cols; j++)
          (*this)(i, j) = A[i][j];
      }
    }
    else
      data.clear();
  }

  void print()
  {
    for (unsigned int i = 0; i < n_rows; i++)
    {
      for (unsigned int j = 0; j < n_cols; j++)
        cout << "\t" << (*this)(i, j);

      cout << endl;
    }
    data.clear();
  }

  inline double& operator()(unsigned int row, unsigned int col)
  {
    return data[row * n_cols + col];
  }
  inline double operator()(unsigned int row, unsigned int col) const
  {
    return data[row * n_cols + col];
  }
};

inline double norm2(const vector<double>& x_vec)
{
  double sum = 0.0;
  for (double e : x_vec)
    sum += e * e;
  return sqrt(sum);
}

enum class StopReason
{
  Undefined,
  MaxGenerations,
  StallAverage,
  StallBest,
  UserRequest
};

class Chronometer
{
protected:
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> Timetype;
  Timetype time_start, time_stop;
  bool initialized;

public:
  Chronometer() : initialized(false)
  {
  }

  void tic()
  {
    initialized = true;
    time_start = std::chrono::high_resolution_clock::now();
  }

  double toc()
  {
    if (!initialized)
      throw runtime_error("Chronometer is not initialized!");
    time_stop = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration<double>(time_stop - time_start).count();
  }
};

template <typename GeneType, typename MiddleCostType>
class Genetic
{
private:
  std::mt19937_64 rng;  // random generator
  std::uniform_real_distribution<double> unif_dist;
  int average_stall_count;
  int best_stall_count;
  vector<double> ideal_objectives;           // for multi-objective
  Matrix extreme_objectives;                 // for multi-objective
  vector<double> scalarized_objectives_min;  // for multi-objective
  Matrix reference_vectors;
  // double shrink_scale;
  unsigned int N_robj;

public:
  typedef ChromosomeType<GeneType, MiddleCostType> thisChromosomeType;
  typedef GenerationType<GeneType, MiddleCostType> thisGenerationType;
  typedef GenerationType_SO_abstract<GeneType, MiddleCostType> thisGenSOAbs;

  ////////////////////////////////////////////////////

  GA_MODE problem_mode;
  unsigned int population;
  double crossover_fraction;
  double mutation_rate;
  bool verbose;
  int generation_step;
  int elite_count;
  int generation_max;
  double tol_stall_average;
  int average_stall_max;
  double tol_stall_best;
  int best_stall_max;
  unsigned int reference_vector_divisions;
  bool enable_reference_vectors;
  bool multi_threading;
  bool dynamic_threading;
  int N_threads;
  bool user_request_stop;
  long idle_delay_us;

  function<void(thisGenerationType&)> calculate_IGA_total_fitness;
  function<double(const thisChromosomeType&)> calculate_SO_total_fitness;
  function<vector<double>(thisChromosomeType&)> calculate_MO_objectives;
  function<vector<double>(const vector<double>&)> distribution_objective_reductions;
  function<void(GeneType&, const function<double(void)>& rnd01)> init_genes;
  function<bool(const GeneType&, MiddleCostType&)> eval_solution;
  function<bool(const GeneType&, MiddleCostType&, const thisGenerationType&)> eval_solution_IGA;
  function<GeneType(const GeneType&, const function<double(void)>& rnd01, double shrink_scale)> mutate;
  function<GeneType(const GeneType&, const GeneType&, const function<double(void)>& rnd01)> crossover;
  function<void(int, const thisGenerationType&, const GeneType&)> SO_report_generation;
  function<void(int, const thisGenerationType&, const vector<unsigned int>&)> MO_report_generation;
  function<void(void)> custom_refresh;
  function<double(int, const function<double(void)>& rnd01)> get_shrink_scale;
  vector<thisGenSOAbs> generations_so_abs;
  thisGenerationType last_generation;

  ////////////////////////////////////////////////////

  Genetic()
    : unif_dist(0.0, 1.0)
    , N_robj(0)
    , problem_mode(GA_MODE::SOGA)
    , population(50)
    , crossover_fraction(0.7)
    , mutation_rate(0.1)
    , verbose(false)
    , generation_step(-1)
    , elite_count(5)
    , generation_max(100)
    , tol_stall_average(1e-4)
    , average_stall_max(10)
    , tol_stall_best(1e-6)
    , best_stall_max(10)
    , reference_vector_divisions(0)
    , enable_reference_vectors(true)
    , multi_threading(true)
    , dynamic_threading(true)
    , N_threads(std::thread::hardware_concurrency())
    , user_request_stop(false)
    , idle_delay_us(1000)
    , calculate_IGA_total_fitness(nullptr)
    , calculate_SO_total_fitness(nullptr)
    , calculate_MO_objectives(nullptr)
    , distribution_objective_reductions(nullptr)
    , init_genes(nullptr)
    , eval_solution(nullptr)
    , eval_solution_IGA(nullptr)
    , mutate(nullptr)
    , crossover(nullptr)
    , SO_report_generation(nullptr)
    , MO_report_generation(nullptr)
    , custom_refresh(nullptr)
    , get_shrink_scale(default_shrink_scale)
  {
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
    rng.seed(ss);
    std::uniform_real_distribution<double> unif(0, 1);
    if (N_threads == 0)  // number of CPU cores not detected.
      N_threads = 8;
  }

  int get_number_reference_vectors(int N_objectives, int N_divisions)
  {
    int m = N_objectives - 1;
    int n = N_divisions;
    if (m > n)
      std::swap(m, n);  // make sure m<=n
    int result = 1;
    for (int k = 1; k <= m; k++)
    {
      result *= m + n - k + 1;
      result /= k;
    }
    return result;
  }

  void calculate_N_robj(const thisGenerationType& g)
  {
    if (!g.chromosomes.size())
      throw runtime_error("Code should not reach here. A87946516564");
    if (distribution_objective_reductions)
      N_robj = (unsigned int)distribution_objective_reductions(g.chromosomes[0].objectives).size();
    else
      N_robj = (unsigned int)g.chromosomes[0].objectives.size();
    if (!N_robj)
      throw runtime_error("Number of the reduced objective is zero");
  }

  void solve_init()
  {
    check_settings();
    // shrink_scale=1.0;
    average_stall_count = 0;
    best_stall_count = 0;
    generation_step = -1;

    if (verbose)
    {
      cout << "**************************************" << endl;
      cout << "*             GA started             *" << endl;
      cout << "**************************************" << endl;
      cout << "population: " << population << endl;
      cout << "elite_count: " << elite_count << endl;
      cout << "crossover_fraction: " << crossover_fraction << endl;
      cout << "mutation_rate: " << mutation_rate << endl;
      cout << "**************************************" << endl;
    }
    Chronometer timer;
    timer.tic();

    thisGenerationType generation0;
    init_population(generation0);
    generation_step = 0;
    finalize_objectives(generation0);

    if (!is_single_objective())
    {
      calculate_N_robj(generation0);
      if (!reference_vector_divisions)
      {
        reference_vector_divisions = 2;
        if (N_robj == 1)
          throw std::runtime_error("The length of objective vector is 1 in a multi-objective optimization");
        while (get_number_reference_vectors(N_robj, reference_vector_divisions + 1) <= (int)population)
          reference_vector_divisions++;
        if (verbose)
        {
          cout << "**************************************" << endl;
          cout << "reference_vector_divisions: " << reference_vector_divisions << endl;
          cout << "**************************************" << endl;
        }
      }
    }
    rank_population(generation0);  // used for ellite tranfre, crossover and mutation
    finalize_generation(generation0);
    if (!is_single_objective())
    {  // muti-objective
      update_ideal_objectives(generation0, true);
      extreme_objectives.clear();
      scalarized_objectives_min.clear();
    }
    generation0.exe_time = timer.toc();
    if (!user_request_stop)
    {
      generations_so_abs.push_back(thisGenSOAbs(generation0));
      report_generation(generation0);
    }
    last_generation = generation0;
  }

  StopReason solve_next_generation()
  {
    Chronometer timer;
    timer.tic();
    generation_step++;
    thisGenerationType new_generation;
    transfer(new_generation);
    crossover_and_mutation(new_generation);

    finalize_objectives(new_generation);
    rank_population(new_generation);  // used for selection
    thisGenerationType selected_generation;
    select_population(new_generation, selected_generation);
    new_generation = selected_generation;
    rank_population(new_generation);  // used for elite tranfre, crossover and mutation
    finalize_generation(new_generation);
    new_generation.exe_time = timer.toc();

    if (!user_request_stop)
    {
      generations_so_abs.push_back(thisGenSOAbs(new_generation));
      report_generation(new_generation);
    }
    last_generation = new_generation;

    return stop_critera();
  }

  StopReason solve()
  {
    StopReason stop = StopReason::Undefined;
    solve_init();
    while (stop == StopReason::Undefined)
      stop = solve_next_generation();
    show_stop_reason(stop);
    return stop;
  }

  std::string stop_reason_to_string(StopReason stop)
  {
    switch (stop)
    {
      case StopReason::Undefined:
        return "No-stop";
        break;
      case StopReason::MaxGenerations:
        return "Maximum generation reached";
        break;
      case StopReason::StallAverage:
        return "Average stalled";
        break;
      case StopReason::StallBest:
        return "Best stalled";
        break;
      case StopReason::UserRequest:
        return "User request";
        break;
      default:
        return "Unknown reason";
    }
  }

protected:
  static double default_shrink_scale(int n_generation, const function<double(void)>& rnd01)
  {
    double scale = (n_generation <= 5 ? 1.0 : 1.0 / sqrt(n_generation - 5 + 1));
    if (rnd01() < 0.4)
      scale *= scale;
    else if (rnd01() < 0.1)
      scale = 1.0;
    return scale;
  }

  double random01()
  {
    return unif_dist(rng);
  }

  void report_generation(const thisGenerationType& new_generation)
  {
    if (is_single_objective())
    {  // SO (including IGA)
      SO_report_generation(generation_step, new_generation,
                           new_generation.chromosomes[new_generation.best_chromosome_index].genes);
    }
    else
    {
      MO_report_generation(generation_step, new_generation, new_generation.fronts[0]);
    }
  }

  void show_stop_reason(StopReason stop)
  {
    if (verbose)
    {
      cout << "Stop criteria: ";
      if (stop == StopReason::Undefined)
        cout << "There is a bug in this function";
      else
        cout << stop_reason_to_string(stop);
      cout << endl;
      cout << "**************************************" << endl;
    }
  }

  void transfer(thisGenerationType& new_generation)
  {
    if (user_request_stop)
      return;

    if (!is_interactive())
    {  // add all members
      for (thisChromosomeType c : last_generation.chromosomes)
        new_generation.chromosomes.push_back(c);
    }
    else
    {
      // in IGA, the final evaluation is expensive
      // therefore, only elites would be transfered.
      for (int i = 0; i < elite_count; i++)
        new_generation.chromosomes.push_back(last_generation.chromosomes[last_generation.sorted_indices[i]]);
    }
  }

  void finalize_generation(thisGenerationType& new_generation)
  {
    if (user_request_stop)
      return;

    if (is_single_objective())
    {
      double best = new_generation.chromosomes[0].total_cost;
      double sum = 0;
      new_generation.best_chromosome_index = 0;

      for (unsigned int i = 0; i < new_generation.chromosomes.size(); i++)
      {
        double current_cost = new_generation.chromosomes[i].total_cost;
        sum += current_cost;
        if (current_cost <= best)
        {
          new_generation.best_chromosome_index = i;
          best = current_cost;
        }
        best = std::min(best, current_cost);
      }

      new_generation.best_total_cost = best;
      new_generation.average_cost = sum / double(new_generation.chromosomes.size());
    }
  }

  void check_settings()
  {
    if (is_interactive())
    {
      if (calculate_IGA_total_fitness == nullptr)
        throw runtime_error("calculate_IGA_total_fitness is null in interactive mode!");
      if (calculate_SO_total_fitness != nullptr)
        throw runtime_error("calculate_SO_total_fitness is not null in interactive mode!");
      if (calculate_MO_objectives != nullptr)
        throw runtime_error("calculate_MO_objectives is not null in interactive mode!");
      if (distribution_objective_reductions != nullptr)
        throw runtime_error("distribution_objective_reductions is not null in interactive mode!");
      if (MO_report_generation != nullptr)
        throw runtime_error("MO_report_generation is not null in interactive mode!");
      if (eval_solution_IGA == nullptr)
        throw runtime_error("eval_solution_IGA is null in interactive mode!");
      if (eval_solution != nullptr)
        throw runtime_error("eval_solution is not null in interactive mode (use eval_solution_IGA instead)!");
    }
    else
    {
      if (calculate_IGA_total_fitness != nullptr)
        throw runtime_error("calculate_IGA_total_fitness is not null in non-interactive mode!");
      if (eval_solution_IGA != nullptr)
        throw runtime_error("eval_solution_IGA is not null in non-interactive mode!");
      if (eval_solution == nullptr)
        throw runtime_error("eval_solution is null!");
      if (is_single_objective())
      {
        if (calculate_SO_total_fitness == nullptr)
          throw runtime_error("calculate_SO_total_fitness is null in single objective mode!");
        if (calculate_MO_objectives != nullptr)
          throw runtime_error("calculate_MO_objectives is not null in single objective mode!");
        if (distribution_objective_reductions != nullptr)
          throw runtime_error("distribution_objective_reductions is not null in single objective mode!");
        if (MO_report_generation != nullptr)
          throw runtime_error("MO_report_generation is not null in single objective mode!");
      }
      else
      {
        if (calculate_SO_total_fitness != nullptr)
          throw runtime_error("calculate_SO_total_fitness is no null in multi-objective mode!");
        if (calculate_MO_objectives == nullptr)
          throw runtime_error("calculate_MO_objectives is null in multi-objective mode!");
        // if(distribution_objective_reductions==nullptr)
        // 	throw runtime_error("distribution_objective_reductions is null in multi-objective mode!");
        if (MO_report_generation == nullptr)
          throw runtime_error("MO_report_generation is null in multi-objective mode!");
      }
    }

    if (init_genes == nullptr)
      throw runtime_error("init_genes is not adjusted.");
    if (mutate == nullptr)
      throw runtime_error("mutate is not adjusted.");
    if (crossover == nullptr)
      throw runtime_error("crossover is not adjusted.");
    if (N_threads < 1)
      throw runtime_error("Number of threads is below 1.");
    if (population < 1)
      throw runtime_error("population is below 1.");
    if (is_single_objective())
    {  // SO (including IGA)
      if (SO_report_generation == nullptr)
        throw runtime_error("SO_report_generation is not adjusted while problem mode is single-objective");
      if (MO_report_generation != nullptr)
        throw runtime_error("MO_report_generation is adjusted while problem mode is single-objective");
    }
    else
    {
      if (SO_report_generation != nullptr)
        throw runtime_error("SO_report_generation is adjusted while problem mode is multi-objective");
      if (MO_report_generation == nullptr)
        throw runtime_error("MO_report_generation is not adjusted while problem mode is multi-objective");
    }
  }

  void select_population(const thisGenerationType& g, thisGenerationType& g2)
  {
    if (user_request_stop)
      return;

    if (is_single_objective())
      select_population_SO(g, g2);
    else
      select_population_MO(g, g2);
  }

  void update_ideal_objectives(const thisGenerationType& g, bool reset)
  {
    if (user_request_stop)
      return;

    if (is_single_objective())
      throw runtime_error("Wrong code A0812473247.");
    if (reset)
    {
      if (distribution_objective_reductions)
        ideal_objectives = distribution_objective_reductions(g.chromosomes[0].objectives);
      else
        ideal_objectives = g.chromosomes[0].objectives;
    }
    unsigned int N_r_objectives = (unsigned int)ideal_objectives.size();
    for (thisChromosomeType x : g.chromosomes)
    {
      vector<double> obj_reduced;
      if (distribution_objective_reductions)
        obj_reduced = distribution_objective_reductions(x.objectives);
      else
        obj_reduced = x.objectives;
      for (unsigned int i = 0; i < N_r_objectives; i++)
        if (obj_reduced[i] < ideal_objectives[i])
          ideal_objectives[i] = obj_reduced[i];
    }
  }

  void select_population_MO(const thisGenerationType& g, thisGenerationType& g2)
  {
    update_ideal_objectives(g, false);
    if (generation_step <= 0)
    {
      g2 = g;
      return;
    }
    g2.chromosomes.clear();
    if (!N_robj)
      throw runtime_error("Number of the reduced objectives is zero. A68756541321");
    const unsigned int N_chromosomes = (unsigned int)g.chromosomes.size();
    Matrix zb_objectives(N_chromosomes, N_robj);
    for (unsigned int i = 0; i < N_chromosomes; i++)
    {
      vector<double> robj_x;
      if (distribution_objective_reductions)
        robj_x = distribution_objective_reductions(g.chromosomes[i].objectives);
      else
        robj_x = g.chromosomes[i].objectives;

      for (unsigned int j = 0; j < N_robj; j++)
        zb_objectives(i, j) = (robj_x[j] - ideal_objectives[j]);
    }
    scalarize_objectives(zb_objectives);
    vector<double> intercepts;
    build_hyperplane_intercepts(intercepts);
    Matrix norm_objectives((unsigned int)g.chromosomes.size(), (unsigned int)intercepts.size());
    for (unsigned int i = 0; i < N_chromosomes; i++)
      for (unsigned int j = 0; j < N_robj; j++)
        norm_objectives(i, j) = zb_objectives(i, j) / intercepts[j];
    if (g.chromosomes.size() == population)
    {
      g2 = g;
      return;
    }
    if (reference_vectors.empty())
    {
      unsigned int obj_dept;
      if (distribution_objective_reductions)
        obj_dept = (unsigned int)distribution_objective_reductions(g.chromosomes[0].objectives).size();
      else
        obj_dept = (unsigned int)g.chromosomes[0].objectives.size();
      reference_vectors = generate_referenceVectors(obj_dept, reference_vector_divisions);
    }
    vector<unsigned int> associated_ref_vector;
    vector<double> distance_ref_vector;

    vector<unsigned int> niche_count;
    Matrix distances;  // row: pop, col: ref_vec
    associate_to_references(g, norm_objectives, associated_ref_vector, distance_ref_vector, niche_count, distances);

    unsigned int last_front_index = 0;
    // select from best fronts as long as they are accommodated in the population
    while (g2.chromosomes.size() + g.fronts[last_front_index].size() <= population)
    {
      for (unsigned int i : g.fronts[last_front_index])
        g2.chromosomes.push_back(g.chromosomes[i]);
      last_front_index++;
    }
    vector<unsigned int> last_front = g.fronts[last_front_index];
    // select randomly from the next front
    vector<unsigned int> to_add;
    while (g2.chromosomes.size() + to_add.size() < population)
    {
      if (!enable_reference_vectors)
      {  // disabling reference points
        unsigned int msz = (unsigned int)last_front.size();
        unsigned int to_add_index = (unsigned int)std::floor(msz * random01());
        if (to_add_index >= msz)
          to_add_index = 0;
        to_add.push_back(last_front[to_add_index]);
        last_front.erase(last_front.begin() + to_add_index);
        continue;
      }

      unsigned int min_niche_index = index_of_min(niche_count);
      vector<unsigned int> min_vec_neighbors;
      for (unsigned int i : last_front)
      {
        if (associated_ref_vector[i] == min_niche_index)
          min_vec_neighbors.push_back(i);
      }
      if (min_vec_neighbors.size() == 0)
      {
        niche_count[min_niche_index] = (unsigned int)(10 * g.chromosomes.size());  // inf
        continue;
      }
      unsigned int next_member_index = 0;  // The assignment is redundant but ok.
      if (niche_count[min_niche_index] == 0)
      {
        double min_val = distances(min_vec_neighbors[0], min_niche_index);
        for (unsigned int i : min_vec_neighbors)
          if (distances(i, min_niche_index) < min_val)
          {
            next_member_index = i;
            min_val = distances(i, min_niche_index);
          }
      }
      else
      {
        unsigned int msz = (unsigned int)min_vec_neighbors.size();
        next_member_index = (unsigned int)(std::floor(msz * random01()));
        if (next_member_index >= msz)
          next_member_index = 0;
      }
      unsigned int to_add_index = min_vec_neighbors[next_member_index];
      to_add.push_back(to_add_index);
      int to_del_front = -1;
      for (unsigned int i = 0; i < last_front.size(); i++)
        if (last_front[i] == to_add_index)
          to_del_front = i;

      if (to_del_front >= 0)
        last_front.erase(last_front.begin() + to_del_front);

      niche_count[min_niche_index]++;
    }
    for (unsigned int i : to_add)
      g2.chromosomes.push_back(g.chromosomes[i]);
  }

  void associate_to_references(const thisGenerationType& gen, const Matrix& norm_objectives,
                               vector<unsigned int>& associated_ref_vector, vector<double>& distance_ref_vector,
                               vector<unsigned int>& niche_count, Matrix& distances)
  {
    unsigned int N_ref = reference_vectors.get_n_rows();
    unsigned int N_x = (unsigned int)gen.chromosomes.size();
    niche_count.assign(N_ref, 0);
    distances.zeros(N_x, N_ref);  // row: pop, col: ref_vec
    associated_ref_vector.assign(gen.chromosomes.size(), 0);
    distance_ref_vector.assign(gen.chromosomes.size(), 0.0);
    for (unsigned int i = 0; i < N_x; i++)
    {
      double dist_min = 0.0;            // to avoid uninitialization warning
      unsigned int dist_min_index = 0;  // to avoid uninitialization warning
      for (unsigned int j = 0; j < N_ref; j++)
      {
        vector<double> reference_vectors_row_j;
        reference_vectors.get_row(j, reference_vectors_row_j);
        double ref_vec_j_norm2 = norm2(reference_vectors_row_j);
        vector<double> w = reference_vectors_row_j;
        for (double& x : w)
          x /= ref_vec_j_norm2;
        vector<double> norm_obj;
        norm_objectives.get_row(i, norm_obj);
        assert(w.size() == norm_obj.size() && "Vector size mismatch! A349687921");
        double scalar_wtnorm = 0.0;
        for (unsigned int k = 0; k < norm_obj.size(); k++)
          scalar_wtnorm += w[k] * norm_obj[k];
        double dist2 = 0.0;
        for (unsigned int k = 0; k < norm_obj.size(); k++)
        {
          double dist_x = norm_obj[k] - scalar_wtnorm * w[k];
          dist2 += dist_x * dist_x;
        }
        double dist = sqrt(dist2);
        distances(i, j) = dist;
        if (j == 0 || dist < dist_min)
        {
          dist_min = dist;
          dist_min_index = j;
        }
      }
      associated_ref_vector[i] = dist_min_index;
      distance_ref_vector[i] = dist_min;
      niche_count[dist_min_index]++;
    }
  }

  void build_hyperplane_intercepts(vector<double>& xinv)
  {
    /*
                        solves A^T*x=[1]
                        y=(1.0)./x
                */
    assert(extreme_objectives.get_n_rows() == extreme_objectives.get_n_cols() && "extreme_objectives must be square! "
                                                                                 "A21658463546");
    int n = extreme_objectives.get_n_rows();
    Matrix L(n, n), U(n, n);
    L.zeros();
    U.zeros();
    for (int i = 0; i < n; i++)
    {
      for (int k = i; k < n; k++)
      {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
          sum += (L(i, j) * U(j, k));
        U(i, k) = extreme_objectives(k, i) - sum;
      }
      for (int k = i; k < n; k++)
      {
        if (i == k)
          L(i, i) = 1;
        else
        {
          double sum = 0.0;
          for (int j = 0; j < i; j++)
            sum += (L(k, j) * U(j, i));
          L(k, i) = (extreme_objectives(i, k) - sum) / U(i, i);
        }
      }
    }
    vector<double> y(n);
    for (int i = 0; i < n; i++)
    {
      double sum = 0.0;
      for (int k = 0; k < i; k++)
        sum += (L(i, k) * y[k]);
      y[i] = (1.0 - sum) / L(i, i);
    }
    vector<double> x(n);
    for (int ii = 0; ii < n; ii++)
    {
      int i = n - 1 - ii;
      double sum = 0.0;
      for (int k = i + 1; k < n; k++)
        sum += (U(i, k) * x[k]);
      x[i] = (y[i] - sum) / U(i, i);
    }
    xinv.resize(n);
    for (int i = 0; i < n; i++)
      xinv[i] = 1.0 / x[i];
  }

  template <typename T>
  unsigned int index_of_min(const vector<T>& v)
  {
    return (unsigned int)(std::distance(v.begin(), std::min_element(v.begin(), v.end())));
  }

  void scalarize_objectives(const Matrix& zb_objectives)
  {
    unsigned int N_objectives = zb_objectives.get_n_cols();
    if (scalarized_objectives_min.empty())
    {
      extreme_objectives.zeros(N_objectives, N_objectives);
      scalarized_objectives_min.assign(N_objectives, std::numeric_limits<double>::infinity());
    }
    for (unsigned int i = 0; i < N_objectives; i++)
    {
      vector<double> w;
      w.assign(N_objectives, 1e-10);
      w[i] = 1.0;
      int Nx = zb_objectives.get_n_rows();
      vector<double> s(Nx);
      for (int j = 0; j < Nx; j++)
      {
        double val_max = -1.0e300;
        for (unsigned int k = 0; k < N_objectives; k++)
          val_max = std::max(val_max, zb_objectives(j, k) / w[k]);
        s[j] = val_max;
      }
      int min_sc_idx = index_of_min(s);
      double min_sc = s[min_sc_idx];

      if (min_sc < scalarized_objectives_min[i])
      {
        scalarized_objectives_min[i] = min_sc;
        for (unsigned int j = 0; j < N_objectives; j++)
          extreme_objectives(i, j) = zb_objectives(min_sc_idx, j);
      }
    }
  }

  void select_population_SO(const thisGenerationType& g, thisGenerationType& g2)
  {
    if (generation_step <= 0)
    {
      g2 = g;
      return;
    }

    if (verbose)
      cout << "Transfered elites: ";
    vector<int> blocked;
    for (int i = 0; i < elite_count; i++)
    {
      g2.chromosomes.push_back(g.chromosomes[g.sorted_indices[i]]);
      blocked.push_back(g.sorted_indices[i]);
      if (verbose)
      {
        cout << (i == 0 ? "" : ", ");
        cout << (g.sorted_indices[i] + 1);
      }
    }
    if (verbose)
      cout << endl;
    for (int i = 0; i < int(population) - elite_count; i++)
    {
      int j;
      bool allowed;
      do
      {
        allowed = true;
        j = select_parent(g);
        for (int k = 0; k < int(blocked.size()) && allowed; k++)
          if (blocked[k] == j)
            allowed = false;
      } while (!allowed);
      g2.chromosomes.push_back(g.chromosomes[j]);
      blocked.push_back(g.sorted_indices[j]);
    }
    if (verbose)
      cout << "Selection done." << endl;
  }

  void rank_population(thisGenerationType& gen)
  {
    if (user_request_stop)
      return;

    if (is_single_objective())
      rank_population_SO(gen);
    else
      rank_population_MO(gen);
  }

  void quicksort_indices_SO(vector<int>& array_indices, const thisGenerationType& gen, int left, int right)
  {
    if (left < right)
    {
      int middle;
      double x = gen.chromosomes[array_indices[left]].total_cost;
      int l = left;
      int r = right;
      while (l < r)
      {
        while ((gen.chromosomes[array_indices[l]].total_cost <= x) && (l < right))
          l++;
        while ((gen.chromosomes[array_indices[r]].total_cost > x) && (r >= left))
          r--;
        if (l < r)
        {
          int temp = array_indices[l];
          array_indices[l] = array_indices[r];
          array_indices[r] = temp;
        }
      }
      middle = r;
      int temp = array_indices[left];
      array_indices[left] = array_indices[middle];
      array_indices[middle] = temp;

      quicksort_indices_SO(array_indices, gen, left, middle - 1);
      quicksort_indices_SO(array_indices, gen, middle + 1, right);
    }
  }

  void rank_population_SO(thisGenerationType& gen)
  {
    int N = int(gen.chromosomes.size());
    gen.sorted_indices.clear();
    gen.sorted_indices.reserve(N);
    for (int i = 0; i < N; i++)
      gen.sorted_indices.push_back(i);

    quicksort_indices_SO(gen.sorted_indices, gen, 0, int(gen.sorted_indices.size()) - 1);

    vector<int> ranks;
    ranks.assign(gen.chromosomes.size(), 0);
    for (unsigned int i = 0; i < gen.chromosomes.size(); i++)
      ranks[gen.sorted_indices[i]] = i;

    generate_selection_chance(gen, ranks);
  }

  void generate_selection_chance(thisGenerationType& gen, const vector<int>& rank)
  {
    double chance_cumulative = 0.0;
    unsigned int N = (unsigned int)gen.chromosomes.size();
    gen.selection_chance_cumulative.clear();
    gen.selection_chance_cumulative.reserve(N);
    for (unsigned int i = 0; i < N; i++)
    {
      chance_cumulative += 1.0 / sqrt(double(rank[i] + 1));
      gen.selection_chance_cumulative.push_back(chance_cumulative);
    }
    for (unsigned int i = 0; i < N; i++)
    {  // normalizing
      gen.selection_chance_cumulative[i] =
          gen.selection_chance_cumulative[i] / gen.selection_chance_cumulative[population - 1];
    }
  }

  void rank_population_MO(thisGenerationType& gen)
  {
    vector<vector<unsigned int>> domination_set;
    vector<int> dominated_count;
    domination_set.reserve(gen.chromosomes.size());
    dominated_count.reserve(gen.chromosomes.size());
    for (unsigned int i = 0; i < gen.chromosomes.size(); i++)
    {
      domination_set.push_back({});
      dominated_count.push_back(0);
    }
    vector<unsigned int> pareto_front;

    for (unsigned int i = 0; i < gen.chromosomes.size(); i++)
    {
      for (unsigned int j = i + 1; j < gen.chromosomes.size(); j++)
      {
        if (dominates(gen.chromosomes[i], gen.chromosomes[j]))
        {
          domination_set[i].push_back(j);
          dominated_count[j]++;
        }
        if (dominates(gen.chromosomes[j], gen.chromosomes[i]))
        {
          domination_set[j].push_back(i);
          dominated_count[i]++;
        }
      }
      if (dominated_count[i] == 0)
        pareto_front.push_back(i);
    }
    gen.fronts.clear();
    gen.fronts.push_back(pareto_front);
    vector<unsigned int> next_front;
    do
    {
      next_front.clear();
      vector<unsigned int>& last_front = gen.fronts[gen.fronts.size() - 1];
      for (unsigned int i : last_front)
        for (unsigned int j : domination_set[i])
          if (--dominated_count[j] == 0)
            next_front.push_back(j);
      if (!next_front.empty())
        gen.fronts.push_back(next_front);
    } while (!next_front.empty());
    vector<int> ranks;
    ranks.assign(gen.chromosomes.size(), 0);
    for (unsigned int i = 0; i < gen.fronts.size(); i++)
      for (unsigned int j = 0; j < gen.fronts[i].size(); j++)
        ranks[gen.fronts[i][j]] = i;
    generate_selection_chance(gen, ranks);
  }

  bool dominates(const thisChromosomeType& a, const thisChromosomeType& b)
  {
    if (a.objectives.size() != b.objectives.size())
      throw runtime_error("vector size mismatch A73592753!");
    for (unsigned int i = 0; i < a.objectives.size(); i++)
      if (a.objectives[i] > b.objectives[i])
        return false;
    for (unsigned int i = 0; i < a.objectives.size(); i++)
      if (a.objectives[i] < b.objectives[i])
        return true;
    return false;
  }

  vector<vector<double>> generate_integerReferenceVectors(int dept, int N_division)
  {
    if (dept < 1)
      throw runtime_error("wrong vector dept!");
    if (dept == 1)
    {
      return { { (double)N_division } };
    }
    vector<vector<double>> result;
    for (int i = 0; i <= N_division; i++)
    {
      vector<vector<double>> tail;
      tail = generate_integerReferenceVectors(dept - 1, N_division - i);

      for (int j = 0; j < int(tail.size()); j++)
      {
        vector<double> v1 = tail[j];
        vector<double> v2(v1.size() + 1);
        v2[0] = i;
        for (int k = 0; k < int(v1.size()); k++)
        {
          v2[k + 1] = v1[k];
        }
        result.push_back(v2);
      }
    }
    return result;
  }

  Matrix generate_referenceVectors(int dept, int N_division)
  {
    Matrix A;
    A = generate_integerReferenceVectors(dept, N_division);
    for (unsigned int i = 0; i < A.get_n_rows(); i++)
      for (unsigned int j = 0; j < A.get_n_cols(); j++)
        A(i, j) /= double(N_division);
    return A;
  }

  bool is_single_objective()
  {
    switch (problem_mode)
    {
      case GA_MODE::SOGA:
        return true;
      case GA_MODE::IGA:
        return true;
      case GA_MODE::NSGA_III:
        return false;
      default:
        throw runtime_error("Code should not reach here!");
    }
  }

  bool is_interactive()
  {
    switch (problem_mode)
    {
      case GA_MODE::SOGA:
        return false;
      case GA_MODE::IGA:
        return true;
      case GA_MODE::NSGA_III:
        return false;
      default:
        throw runtime_error("Code should not reach here!");
    }
  }

  void init_population_range(thisGenerationType* p_generation0, int index_begin, int index_end, unsigned int* attemps,
                             int* active_thread)
  {
    int dummy;
    for (int i = index_begin; i <= index_end; i++)
      init_population_single(p_generation0, i, attemps, &dummy);
    *active_thread = 0;  // false
  }

  void init_population_single(thisGenerationType* p_generation0, int index, unsigned int* attemps, int* active_thread)
  {
    bool accepted = false;
    while (!accepted)
    {
      thisChromosomeType X;
      init_genes(X.genes, [this]() { return random01(); });
      if (is_interactive())
      {
        if (eval_solution_IGA(X.genes, X.middle_costs, *p_generation0))
        {
          // in IGA mode, code cannot run in parallel.
          p_generation0->chromosomes.push_back(X);
          accepted = true;
        }
      }
      else
      {
        if (eval_solution(X.genes, X.middle_costs))
        {
          if (index >= 0)
            p_generation0->chromosomes[index] = X;
          else
            p_generation0->chromosomes.push_back(X);
          accepted = true;
        }
      }
      (*attemps)++;
    }
    *active_thread = 0;  // false
  }

  void idle()
  {
    if (custom_refresh != nullptr)
      custom_refresh();
    if (idle_delay_us > 0)
      std::this_thread::sleep_for(std::chrono::microseconds(idle_delay_us));
  }

  void init_population(thisGenerationType& generation0)
  {
    generation0.chromosomes.clear();

    unsigned int total_attempts = 0;
    if (!multi_threading || N_threads == 1 || is_interactive())
    {
      int dummy;
      for (unsigned int i = 0; i < population && !user_request_stop; i++)
        init_population_single(&generation0, -1, &total_attempts, &dummy);
    }
    else
    {
      for (unsigned int i = 0; i < population; i++)
        generation0.chromosomes.push_back(thisChromosomeType());
      vector<int> active_threads;  // vector<bool> is broken
      active_threads.assign(N_threads, 0);
      vector<unsigned int> attempts;
      attempts.assign(N_threads, 0);

      vector<std::thread> thread_pool;
      for (int i = 0; i < N_threads; i++)
        thread_pool.push_back(std::thread());
      for (std::thread& th : thread_pool)
        if (th.joinable())
          th.join();

      if (dynamic_threading)
      {
        unsigned int x_index = 0;
        while (x_index < population && !user_request_stop)
        {
          int free_thread = -1;
          for (int i = 0; i < N_threads && free_thread < 0; i++)
          {
            if (!active_threads[i])
            {
              free_thread = i;
              if (thread_pool[free_thread].joinable())
                thread_pool[free_thread].join();
            }
          }
          if (free_thread > -1)
          {
            active_threads[free_thread] = 1;
            thread_pool[free_thread] =
                std::thread(&std::remove_reference<decltype(*this)>::type::init_population_single, this, &generation0,
                            int(x_index), &attempts[free_thread], &(active_threads[free_thread]));
            x_index++;
          }
          else
            idle();
        }  // while
      }    // endif: dynamic threading
      else
      {  // on static threading
        int x_index_start = 0;
        int x_index_end = 0;
        int pop_chunk = population / N_threads;
        pop_chunk = std::max(pop_chunk, 1);
        for (int i = 0; i < N_threads; i++)
        {
          x_index_end = x_index_start + pop_chunk;
          if (i + 1 == N_threads)  // last chunk
            x_index_end = population - 1;
          else
            x_index_end = std::min(x_index_end, int(population) - 1);

          if (x_index_end >= x_index_start)
          {
            active_threads[i] = 1;
            thread_pool[i] = std::thread(&std::remove_reference<decltype(*this)>::type::init_population_range, this,
                                         &generation0, x_index_start, x_index_end, &attempts[i], &(active_threads[i]));
          }
          x_index_start = x_index_end + 1;
        }
      }  // endif: static threading

      bool all_tasks_finished;
      do
      {
        all_tasks_finished = true;
        for (int i = 0; i < N_threads; i++)
          if (active_threads[i])
            all_tasks_finished = false;
        if (!all_tasks_finished)
          idle();
      } while (!all_tasks_finished);
      // wait for tasks to finish
      for (std::thread& th : thread_pool)
        if (th.joinable())
          th.join();

      for (unsigned int ac : attempts)
        total_attempts += ac;
    }

    /////////////////////

    if (verbose)
    {
      cout << "Initial population of " << population << " was created with " << total_attempts << " attemps." << endl;
    }
  }

  int select_parent(const thisGenerationType& g)
  {
    int N_max = int(g.chromosomes.size());
    double r = random01();
    int position = 0;
    while (position < N_max && g.selection_chance_cumulative[position] < r)
      position++;
    return position;
  }

  void crossover_and_mutation_range(thisGenerationType* p_new_generation, unsigned int pop_previous_size,
                                    int x_index_begin, int x_index_end, int* active_thread)
  {
    int dummy;
    for (int i = x_index_begin; i <= x_index_end; i++)
      crossover_and_mutation_single(p_new_generation, pop_previous_size, i, &dummy);
    *active_thread = 0;  // false
  }

  void crossover_and_mutation_single(thisGenerationType* p_new_generation, unsigned int pop_previous_size, int index,
                                     int* active_thread)
  {
    if (verbose)
      cout << "Action: crossover" << endl;

    bool successful = false;
    while (!successful)
    {
      thisChromosomeType X;

      int pidx_c1 = select_parent(last_generation);
      int pidx_c2 = select_parent(last_generation);
      if (pidx_c1 == pidx_c2)
        continue;
      if (verbose)
        cout << "Crossover of chromosomes " << pidx_c1 << "," << pidx_c2 << endl;
      GeneType Xp1 = last_generation.chromosomes[pidx_c1].genes;
      GeneType Xp2 = last_generation.chromosomes[pidx_c2].genes;
      X.genes = crossover(Xp1, Xp2, [this]() { return random01(); });
      if (random01() <= mutation_rate)
      {
        if (verbose)
          cout << "Mutation of chromosome " << endl;
        double shrink_scale = get_shrink_scale(generation_step, [this]() { return random01(); });
        X.genes = mutate(
            X.genes, [this]() { return random01(); }, shrink_scale);
      }
      if (is_interactive())
      {
        if (eval_solution_IGA(X.genes, X.middle_costs, *p_new_generation))
        {
          p_new_generation->chromosomes.push_back(X);
          successful = true;
        }
      }
      else
      {
        if (eval_solution(X.genes, X.middle_costs))
        {
          if (index >= 0)
            p_new_generation->chromosomes[pop_previous_size + index] = X;
          else
            p_new_generation->chromosomes.push_back(X);
          successful = true;
        }
      }
    }
    *active_thread = 0;  // false
  }

  void crossover_and_mutation(thisGenerationType& new_generation)
  {
    if (user_request_stop)
      return;

    if (crossover_fraction <= 0.0 || crossover_fraction > 1.0)
      throw runtime_error("Wrong crossover fractoin");
    if (mutation_rate < 0.0 || mutation_rate > 1.0)
      throw runtime_error("Wrong mutation rate");
    if (generation_step <= 0)
      return;
    unsigned int N_add = (unsigned int)(std::round(double(population) * (crossover_fraction)));
    unsigned int pop_previous_size = (unsigned int)new_generation.chromosomes.size();
    if (is_interactive())
    {
      if (N_add + elite_count != population)
        throw runtime_error("In IGA mode, elite fraction + crossover fraction must be equal to 1.0 !");
    }

    if (!multi_threading || N_threads == 1 || is_interactive())
    {
      int dummy;
      for (unsigned int i = 0; i < N_add && !user_request_stop; i++)
        crossover_and_mutation_single(&new_generation, pop_previous_size, -1, &dummy);
    }
    else
    {
      for (unsigned int i = 0; i < N_add; i++)
        new_generation.chromosomes.push_back(thisChromosomeType());
      vector<int> active_threads;  // vector<bool> is broken
      active_threads.assign(N_threads, 0);

      vector<std::thread> thread_pool;
      for (int i = 0; i < N_threads; i++)
        thread_pool.push_back(std::thread());
      for (std::thread& th : thread_pool)
        if (th.joinable())
          th.join();

      if (dynamic_threading)
      {
        unsigned int x_index = 0;
        while (x_index < N_add && !user_request_stop)
        {
          int free_thread = -1;
          for (int i = 0; i < N_threads && free_thread < 0; i++)
          {
            if (!active_threads[i])
            {
              free_thread = i;
              if (thread_pool[free_thread].joinable())
                thread_pool[free_thread].join();
            }
          }
          if (free_thread > -1)
          {
            active_threads[free_thread] = 1;
            thread_pool[free_thread] =
                std::thread(&std::remove_reference<decltype(*this)>::type::crossover_and_mutation_single, this,
                            &new_generation, pop_previous_size, int(x_index), &(active_threads[free_thread]));
            x_index++;
          }
          else
            idle();
        }
      }  // endif: dynamic threading
      else
      {  // on static threading
        int x_index_start = 0;
        int x_index_end = 0;
        int pop_chunk = N_add / N_threads;
        pop_chunk = std::max(pop_chunk, 1);
        for (int i = 0; i < N_threads; i++)
        {
          x_index_end = x_index_start + pop_chunk;
          if (i + 1 == N_threads)  // last chunk
            x_index_end = N_add - 1;
          else
            x_index_end = std::min(x_index_end, int(N_add) - 1);

          if (x_index_end >= x_index_start)
          {
            active_threads[i] = 1;

            thread_pool[i] =
                std::thread(&std::remove_reference<decltype(*this)>::type::crossover_and_mutation_range, this,
                            &new_generation, pop_previous_size, x_index_start, x_index_end, &(active_threads[i]));
          }
          x_index_start = x_index_end + 1;
        }
      }  // endif: static threading

      bool all_tasks_finished;
      do
      {
        all_tasks_finished = true;
        for (int i = 0; i < N_threads; i++)
          if (active_threads[i])
            all_tasks_finished = false;
        if (!all_tasks_finished)
          idle();
      } while (!all_tasks_finished);

      // wait for tasks to finish
      for (std::thread& th : thread_pool)
        if (th.joinable())
          th.join();
    }
  }

  StopReason stop_critera()
  {
    if (generation_step < 2 && !user_request_stop)
      return StopReason::Undefined;

    if (is_single_objective())
    {
      const thisGenSOAbs& g1 = generations_so_abs[int(generations_so_abs.size()) - 2];
      const thisGenSOAbs& g2 = generations_so_abs[int(generations_so_abs.size()) - 1];

      if (std::abs(g1.best_total_cost - g2.best_total_cost) < tol_stall_best)
        best_stall_count++;
      else
        best_stall_count = 0;
      if (std::abs(g1.average_cost - g2.average_cost) < tol_stall_average)
        average_stall_count++;
      else
        average_stall_count = 0;
    }

    if (generation_step >= generation_max)
      return StopReason::MaxGenerations;

    if (average_stall_count >= average_stall_max)
      return StopReason::StallAverage;

    if (best_stall_count >= best_stall_max)
      return StopReason::StallBest;

    if (user_request_stop)
      return StopReason::UserRequest;

    return StopReason::Undefined;
  }

  void finalize_objectives(thisGenerationType& g)
  {
    if (user_request_stop)
      return;

    switch (problem_mode)
    {
      case GA_MODE::SOGA:
        for (int i = 0; i < int(g.chromosomes.size()); i++)
          g.chromosomes[i].total_cost = calculate_SO_total_fitness(g.chromosomes[i]);
        break;
      case GA_MODE::IGA:
        calculate_IGA_total_fitness(g);
        break;
      case GA_MODE::NSGA_III:
        for (unsigned int i = 0; i < g.chromosomes.size(); i++)
          g.chromosomes[i].objectives = calculate_MO_objectives(g.chromosomes[i]);
        break;
      default:
        throw runtime_error("Code should not reach here!");
    }
  }
};

NS_EA_END
