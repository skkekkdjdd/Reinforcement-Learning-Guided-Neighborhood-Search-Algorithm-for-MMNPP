#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <string>
#include <set>
#include <random>
#include <chrono>
#include <climits>
#include "problem.h"

using namespace std;

problem_struct* problem;

// Algorithm time limit (in seconds)
double alg_time_limit = 1800;
double cur_time = 0.0;

bool shake_flag = false;

// Store the local optimum solutions and their iters
bool PreviousEncounter;

int no_more_impr = 0;
int no_more_impr_max = 0;
int no_more_impr_last = 0;

// Macro to access the weight matrix
#define weight_mat(i,j) problem->weight[(i)*problem->m + j]

default_random_engine random_engine;
random_device random_seed;

uniform_real_distribution<float> e_greedy_random_generator(0.0, 1.0);

int Q_op = 0;

vector<string> statistic_array(101);
vector<int> time_array(101);
int statistic_index = 1;

class HistoryBest {
public:
	HistoryBest(int capacity = 20) : capacity_(capacity), current_size_(0), insert_index_(0) {
		best_solutions_.reserve(capacity);
		iteration_rounds_.reserve(capacity);
	}

	void insert(double solution, int iteration) {
		auto it = solution_map_.find(solution);

		if (it != solution_map_.end()) {
			SolutionInfo& info = it->second;
			int previous_iteration = info.previous_iteration;
			int index = info.index;
			int iteration_diff = iteration - previous_iteration;
			last_iteration_diff_ = iteration_diff;

			iteration_rounds_[index] = iteration;
			info.previous_iteration = iteration;
		}
		else {
			if (current_size_ < capacity_) {
				best_solutions_.push_back(solution);
				iteration_rounds_.push_back(iteration);
				solution_map_[solution] = { current_size_, iteration };
				++current_size_;
			}
			else {
				int replace_index = insert_index_ % capacity_;
				int old_solution = best_solutions_[replace_index];

				solution_map_.erase(old_solution);

				best_solutions_[replace_index] = solution;
				iteration_rounds_[replace_index] = iteration;
				solution_map_[solution] = { replace_index, iteration };

				++insert_index_;
			}
		}
	}

	bool contains(int solution) const {
		return solution_map_.find(solution) != solution_map_.end();
	}

	int getIterationDifference() const {
		return last_iteration_diff_;
	}

private:
	struct SolutionInfo {
		int index;
		int previous_iteration;
	};

	int capacity_;
	int current_size_;
	int insert_index_;
	vector<int> best_solutions_;
	vector<int> iteration_rounds_;
	unordered_map<int, SolutionInfo> solution_map_;
	int last_iteration_diff_;
};

void rearrange(std::vector<int>& solution)
{
	// Create a new vector to store the rearranged solution
	std::vector<int> rearrange(problem->n, -1);
	int index = 0;
	// Assign new indices to partitions in the solution
	for (int i = 0; i < problem->n; ++i) {
		int px = solution[i];
		if (rearrange[i] == -1) {
			rearrange[i] = index;
			for (int j = i + 1; j < problem->n; ++j) {
				if (solution[j] == px) {
					rearrange[j] = index;
				}
			}
			index++;
		}
	}
	// Update the solution with rearranged indices
	index = 0;
	for (auto x : rearrange) {
		solution[index] = x;
		index++;
	}
}

std::vector<int> RandomizedGenerator() {
	// Initialize solution vector
	std::vector<int> solution(problem->n, -1);
	std::default_random_engine random_engine;
	random_engine.seed(time(0));

	int kx = 0;
	// Ensure feasibility 
	while (kx < problem->k) {
		int k = random_engine() % problem->n;
		// Find an unassigned element
		while (solution[k] != -1) {
			k = random_engine() % problem->n;
		}
		solution[k] = kx;
		kx++;
	}
	// Assign remaining elements randomly
	for (auto i = 0; i < problem->n; ++i) {
		if (solution[i] == -1) {
			int part_r = random_engine() % problem->k;
			solution[i] = part_r;
		}
	}
	rearrange(solution);
	return solution;
}

// Calculate the maximum difference between max and min values in a specific dimension l
double max_minus_min(std::vector<std::vector<double>>& sum, int l) {
	double max_value = -10000000.0; double min_value = 1000000000.0;
	for (int j = 0; j < problem->k; ++j) {
		if (max_value < sum[j][l])
			max_value = sum[j][l];
	}
	for (int j = 0; j < problem->k; ++j) {
		if (min_value > sum[j][l])
			min_value = sum[j][l];
	}
	return max_value - min_value;
}

// Objective function to calculate the fitness of a solution
double objective(std::vector<int>& solution) {
	// Return maximum integer value for empty solutions
	if (solution.empty())
		return INT_MAX;
	double value = 0.0;
	std::vector<std::vector<double>> sum;  // sum[i][j]: sum of coordinate j from {0,...,m-1} of vectors in partition i 

	for (int i = 0; i < problem->k; ++i) {
		std::vector<double> sum_i(problem->m, 0.0);  // Initialize sum vector for each partition
		sum.push_back(sum_i);
	}
	int index = 0;
	for (int dim : solution) {
		for (int j = 0; j < problem->m; ++j)
			// Add weights to the corresponding partition
			sum[dim][j] += weight_mat(index, j);
		index++; // next vector
	}

	double obj_val = -INT_MAX;

	for (int j = 0; j < problem->m; ++j) {
		double max_min = max_minus_min(sum, j);
		obj_val = (std::max)(obj_val, max_min);
	}
	return obj_val;
}

// Perform a move operation in the shake process
void shake_move(vector<int>& sol, int i, int p) {
	sol[i] = p;
}

// Shake swap function to swap two elements in the solution
void shake_swap(vector<int>& sol, int i, int j) {
	int tmp = sol[i];
	sol[i] = sol[j];
	sol[j] = tmp;
}

// 1-move operation during local search
double move_1(vector<int>& sol, int i, int p, vector<vector<double>>& p_sum) {
	for (int j = 0; j < problem->m; j++) {
		// Subtract weight from the original partition
		p_sum[sol[i]][j] -= weight_mat(i, j);
		// Add weight to the new partition
		p_sum[p][j] += weight_mat(i, j);
	}
	double new_obj_value = 0;

	// Calculate the difference
	for (int z = 1; z < problem->k; z++) {
		for (int j = 0; j < z; j++) {
			for (int q = 0; q < problem->m; q++) {
				double diff = abs(p_sum[z][q] - p_sum[j][q]);
				if (diff > new_obj_value) {
					new_obj_value = diff;
				}
			}
		}
	}
	// Revert the weight changes
	for (int j = 0; j < problem->m; j++) {
		p_sum[sol[i]][j] += weight_mat(i, j);
		p_sum[p][j] -= weight_mat(i, j);
	}
	return new_obj_value;
}

// Swap function for 2-swap operation
double swap_2(std::vector<int>& sol, int i, int j, vector<vector<double>>& p_sum) {
	for (int s = 0; s < problem->m; s++) {
		p_sum[sol[i]][s] -= weight_mat(i, s);
		p_sum[sol[j]][s] -= weight_mat(j, s);
		p_sum[sol[i]][s] += weight_mat(j, s);
		p_sum[sol[j]][s] += weight_mat(i, s);
	}
	double new_obj_val = 0;
	for (int z = 1; z < problem->k; z++) {
		for (int j = 0; j < z; j++)
			for (int q = 0; q < problem->m; q++) {
				double diff = abs(p_sum[z][q] - p_sum[j][q]);
				if (diff > new_obj_val)
					new_obj_val = diff;
			}
	}
	for (int s = 0; s < problem->m; s++) {
		p_sum[sol[i]][s] += weight_mat(i, s);
		p_sum[sol[j]][s] += weight_mat(j, s);
		p_sum[sol[i]][s] -= weight_mat(j, s);
		p_sum[sol[j]][s] -= weight_mat(i, s);
	}
	return new_obj_val;
}

// Swap function for 3-swap operation
double swap_3(std::vector<int>& sol, int i, int j, int q, vector<vector<double>>& p_sum) {
	for (int s = 0; s < problem->m; s++) {
		p_sum[sol[i]][s] -= weight_mat(i, s);
		p_sum[sol[j]][s] -= weight_mat(j, s);
		p_sum[sol[q]][s] -= weight_mat(q, s);
		p_sum[sol[j]][s] += weight_mat(i, s);
		p_sum[sol[q]][s] += weight_mat(j, s);
		p_sum[sol[i]][s] += weight_mat(q, s);
	}
	double new_obj_value = 0;
	for (int z = 1; z < problem->k; z++) {
		for (int j = 0; j < z; j++)
			for (int r = 0; r < problem->m; r++) {
				double diff = abs(p_sum[z][r] - p_sum[j][r]);
				if (diff > new_obj_value)
					new_obj_value = diff;
			}
	}
	for (int s = 0; s < problem->m; s++) {
		p_sum[sol[i]][s] += weight_mat(i, s);
		p_sum[sol[j]][s] += weight_mat(j, s);
		p_sum[sol[q]][s] += weight_mat(q, s);
		p_sum[sol[j]][s] -= weight_mat(i, s);
		p_sum[sol[q]][s] -= weight_mat(j, s);
		p_sum[sol[i]][s] -= weight_mat(q, s);
	}
	return new_obj_value;
}

// Shake function to escape local optima
void shake(vector<int>& sol, unordered_map<int, int>& maps_count, int& intense_shake_magnitude) {
	// Set up the random number generator
	random_engine.seed(random_seed() + clock());
	for (int i = intense_shake_magnitude + 1; i >= 0; i--) {
		while (true) {
			int shake_i = random_engine() % problem->n;
			int shake_p = random_engine() % problem->k;
			while (shake_p == sol[shake_i]) {
				shake_p = random_engine() % problem->k;
			}
			if (maps_count[sol[shake_i]] > 1) {  // Ensure the element is not alone in its partition
				shake_move(sol, shake_i, shake_p);
				break;
			}
		}
	}
	double obj_val = objective(sol);
}

// Local Search function
double LS_First(std::vector<int>& sol, std::unordered_map<int, int> maps_count, int r, vector<vector<double>>& p_sum) {
	// Set up the random number generator
	random_engine.seed(random_seed() + clock());

	int n = problem->n;
	int k = problem->k;
	int m = problem->m;

	// Initialize the partial sum array for each partition
	for (int i = 0; i < k; ++i) {
		fill(p_sum[i].begin(), p_sum[i].end(), 0.0);
	}

	// Initial objective value
	double obj_val = objective(sol);

	// Initialize the partial sum array with the current solution
	for (int i = 0; i < problem->n; i++)
		for (int j = 0; j < problem->m; j++)
			p_sum[sol[i]][j] += weight_mat(i, j);

	int impr = 1;	// Improvement flag
	while (impr) {
		while (impr) {
			impr = 0;
			int best_i = -1;
			int best_p = -1;
			double best_obj_val = obj_val;

			// 1-move operation
			for (int i = random_engine() % problem->n, cnt = 0; cnt < n; cnt++) {
				int selected_vector = (i + cnt) % problem->n;
				// Skip if the element is alone in its partition
				if (maps_count[sol[selected_vector]] == 1)
					continue;
				for (int p = random_engine() % problem->k, cnt_p = 0; cnt_p < k; cnt_p++) {
					int selected_set = (p + cnt_p) % problem->k;
					// Skip if the partition is the same
					if (selected_set == sol[selected_vector])
						continue;
					double new_obj_val = move_1(sol, selected_vector, selected_set, p_sum);
					// Check for improvement
					if (new_obj_val < best_obj_val - 0.1) {
						best_i = selected_vector;
						best_p = selected_set;
						best_obj_val = new_obj_val;
						impr = 1;
						break;
					}
				}
			}
			if (impr) {
				maps_count[sol[best_i]]--;
				maps_count[best_p]++;
				int tmp_ori_set = sol[best_i];
				// Update partial sums
				for (int j = 0; j < problem->m; j++) {
					p_sum[sol[best_i]][j] -= weight_mat(best_i, j);
					p_sum[best_p][j] += weight_mat(best_i, j);
				}
				sol[best_i] = best_p;
				obj_val = best_obj_val;
				impr = 1;
			}
		}
		if (r >= 2) {
			int best_i = -1;
			int best_j = -1;
			double best_obj_val = obj_val;
			// 2-swap operation
			for (int i = random_engine() % problem->n, cnt = 0; cnt < n; cnt++) {
				int selected_vector = (i + cnt) % problem->n;
				if (selected_vector == 0) {
					continue;
				}
				for (int j = random_engine() % selected_vector, cnt_j = 0; cnt_j < selected_vector; cnt_j++) {
					int selected_vector_2 = (j + cnt_j) % selected_vector;
					if (sol[selected_vector] == sol[selected_vector_2])
						continue;
					double new_obj_val = swap_2(sol, selected_vector, selected_vector_2, p_sum);
					if (new_obj_val < best_obj_val - 0.1) {
						best_obj_val = new_obj_val;
						best_i = selected_vector;
						best_j = selected_vector_2;
						impr = 1;
						break;
					}
				}
			}
			if (impr) {
				for (int s = 0; s < problem->m; s++) {
					p_sum[sol[best_i]][s] -= weight_mat(best_i, s);
					p_sum[sol[best_j]][s] -= weight_mat(best_j, s);
					p_sum[sol[best_i]][s] += weight_mat(best_j, s);
					p_sum[sol[best_j]][s] += weight_mat(best_i, s);
				}
				int pi = sol[best_i];
				sol[best_i] = sol[best_j];
				sol[best_j] = pi;
				obj_val = best_obj_val;
				impr = 1;
			}
		}
		// LS3 find (i, j, q) and interchange their parts
		if (r >= 3)
		{
			int best_i = -1;
			int best_j = -1;
			int best_q = -1;
			double best_obj_val = obj_val;

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (sol[i] == sol[j])
						continue;
					for (int q = 0; q < n; q++) {
						if (sol[q] == sol[i] || sol[q] == sol[j])
							continue;
						// when k=2 this search does not make sense -- because the next command is never achieved.
						double new_obj_val = swap_3(sol, i, j, q, p_sum);
						if (new_obj_val < best_obj_val - 0.1) {
							best_obj_val = new_obj_val;
							best_i = i;
							best_j = j;
							best_q = q;
							impr = 1;
							break;
						}
					}
				}
			}
			if (impr) {
				// i gets what j has
				// j gets what q has
				// q gets what i has (no changes in maps_count structure)
				for (int s = 0; s < problem->m; s++) {
					p_sum[sol[best_i]][s] -= weight_mat(best_i, s);
					p_sum[sol[best_j]][s] -= weight_mat(best_j, s);
					p_sum[sol[best_q]][s] -= weight_mat(best_q, s);
					p_sum[sol[best_j]][s] += weight_mat(best_i, s);
					p_sum[sol[best_q]][s] += weight_mat(best_j, s);
					p_sum[sol[best_i]][s] += weight_mat(best_q, s);
				}
				int pi = sol[best_i];
				sol[best_i] = sol[best_j];
				sol[best_j] = sol[best_q];
				sol[best_q] = pi;
				obj_val = best_obj_val;
				impr = 1;
				continue;
			}
		}
	}
	return obj_val;
}

// Local Search function
double LS(std::vector<int>& sol, std::unordered_map<int, int> maps_count, int r, vector<vector<double>>& p_sum) {
	int n = problem->n;
	int k = problem->k;
	int m = problem->m;

	// Initialize the partial sum array for each partition
	for (int i = 0; i < k; ++i) {
		fill(p_sum[i].begin(), p_sum[i].end(), 0.0);
	}
	cout << "fill" << endl;
	cout << "psum[0][0] : " << p_sum[0][0] << endl;

	// Initial objective value
	double obj_val = objective(sol);

	// Initialize the partial sum array with the current solution
	for (int i = 0; i < problem->n; i++)
		for (int j = 0; j < problem->m; j++) {
			cout << "i : " << i << " j : " << j << endl;
			cout << "sol[i] : " << sol[i] << endl;
			cout << "p_sum[sol[i]][j] : " << p_sum[sol[i]][j] << endl;
			p_sum[sol[i]][j] += weight_mat(i, j);
		}


	cout << "weight" << endl;

	int impr = 1;	// Improvement flag
	while (impr) {
		while (impr) {
			impr = 0;
			int best_i = -1;
			int best_p = -1;
			double best_obj_val = obj_val;

			// 1-move operation
			for (int i = 0; i < n; i++) {
				// Skip if the element is alone in its partition
				if (maps_count[sol[i]] == 1)
					continue;
				for (int p = 0; p < k; p++) {
					// Skip if the partition is the same
					if (p == sol[i])
						continue;
					double new_obj_val = move_1(sol, i, p, p_sum);
					// Check for improvement
					if (new_obj_val < best_obj_val - 0.1) {
						best_i = i;
						best_p = p;
						best_obj_val = new_obj_val;
						impr = 1;
					}
				}
			}
			if (impr) {
				maps_count[sol[best_i]]--;
				maps_count[best_p]++;
				int tmp_ori_set = sol[best_i];
				// Update partial sums
				for (int j = 0; j < problem->m; j++) {
					p_sum[sol[best_i]][j] -= weight_mat(best_i, j);
					p_sum[best_p][j] += weight_mat(best_i, j);
				}
				sol[best_i] = best_p;
				obj_val = best_obj_val;
				impr = 1;
			}
		}
		if (r >= 2) {
			int best_i = -1;
			int best_j = -1;
			double best_obj_val = obj_val;
			// 2-swap operation
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < i; j++) {
					if (sol[i] == sol[j])
						continue;
					double new_obj_val = swap_2(sol, i, j, p_sum);
					if (new_obj_val < best_obj_val - 0.1) {
						best_obj_val = new_obj_val;
						best_i = i;
						best_j = j;
						impr = 1;
					}
				}
			}
			if (impr) {
				for (int s = 0; s < problem->m; s++) {
					p_sum[sol[best_i]][s] -= weight_mat(best_i, s);
					p_sum[sol[best_j]][s] -= weight_mat(best_j, s);
					p_sum[sol[best_i]][s] += weight_mat(best_j, s);
					p_sum[sol[best_j]][s] += weight_mat(best_i, s);
				}
				int pi = sol[best_i];
				sol[best_i] = sol[best_j];
				sol[best_j] = pi;
				obj_val = best_obj_val;
				impr = 1;
		}
		// LS3 find (i, j, q) and interchange their parts
		if (r >= 3)
		{
			int best_i = -1;
			int best_j = -1;
			int best_q = -1;
			double best_obj_val = obj_val;

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (sol[i] == sol[j])
						continue;
					for (int q = 0; q < n; q++) {
						if (sol[q] == sol[i] || sol[q] == sol[j])
							continue;
						// when k=2 this search does not make sense -- because the next command is never achieved.
						double new_obj_val = swap_3(sol, i, j, q, p_sum);
						if (new_obj_val < best_obj_val - 0.1) {
							best_obj_val = new_obj_val;
							best_i = i;
							best_j = j;
							best_q = q;
							impr = 1;
						}
					}
				}
			}
			if (impr) {
				// i gets what j has
				// j gets what q has
				// q gets what i has (no changes in maps_count structure)
				for (int s = 0; s < problem->m; s++) {
					p_sum[sol[best_i]][s] -= weight_mat(best_i, s);
					p_sum[sol[best_j]][s] -= weight_mat(best_j, s);
					p_sum[sol[best_q]][s] -= weight_mat(best_q, s);
					p_sum[sol[best_j]][s] += weight_mat(best_i, s);
					p_sum[sol[best_q]][s] += weight_mat(best_j, s);
					p_sum[sol[best_i]][s] += weight_mat(best_q, s);
				}
				int pi = sol[best_i];
				sol[best_i] = sol[best_j];
				sol[best_j] = sol[best_q];
				sol[best_q] = pi;
				obj_val = best_obj_val;
				impr = 1;
				continue;
			}
		}
	}
	return obj_val;
}

// Local Search function
double LS_Qvalue(vector<int>& sol, unordered_map<int, int> maps_count, int &r, double epsilon,
	vector<vector<double>>& p_sum, vector<vector<int>>& Q_move, vector<vector<int>>& Q_swap,
	double para_alpha, double para_gamma) {
	// Set up the random number generator
	random_engine.seed(random_seed() + clock());

	int n = problem->n;
	int k = problem->k;
	int m = problem->m;

	// Initialize the partial sum array for each partition
	for (int i = 0; i < k; ++i) {
		fill(p_sum[i].begin(), p_sum[i].end(), 0.0);
	}

	// Initial objective value
	double obj_val = objective(sol);

	// Initialize the partial sum array with the current solution
	for (int i = 0; i < problem->n; i++)
		for (int j = 0; j < problem->m; j++)
			p_sum[sol[i]][j] += weight_mat(i, j);

	int impr = 1;	// Improvement flag
	double reward = 0;
	while (impr) {
		while (impr) {
			impr = 0;
			int best_i = -1;
			int best_p = -1;
			double best_obj_val = obj_val;

			// 1-move operation
			for (int i = random_engine() % problem->n, cnt = 0; cnt < n; cnt++) {
				int selected_vector = (i + cnt) % problem->n;
				// Skip if the element is alone in its partition
				if (maps_count[sol[selected_vector]] == 1)
					continue;
				float e_greedy_random_para = e_greedy_random_generator(random_engine);
				// Random
				if (e_greedy_random_para <= epsilon) {
					for (int p = random_engine() % problem->k, cnt_p = 0; cnt_p < k; cnt_p++) {
						int selected_set = (p + cnt_p) % problem->k;
						// Skip if the partition is the same
						if (selected_set == sol[selected_vector])
							continue;
						double new_obj_val = move_1(sol, selected_vector, selected_set, p_sum);
						// Check for improvement
						if (new_obj_val < best_obj_val - 0.1) {
							best_i = selected_vector;
							best_p = selected_set;
							best_obj_val = new_obj_val;
							impr = 1;
							break;
						}
					}
					// According to Q-value matrix
				}
				else {
					int max_Qvalue_index = -1;
					int max_Qvalue = -1;
					int equ_cnt = 1;
					int Qi;
					for (Qi = 0; Qi < problem->k; Qi++) {
						if (Q_swap[selected_vector][Qi] > max_Qvalue + 0.01 && sol[Qi] != sol[selected_vector]) {
							max_Qvalue = Q_move[selected_vector][Qi];
							max_Qvalue_index = Qi;
						}
						else if (Q_swap[selected_vector][Qi] < max_Qvalue + 0.01 && Q_swap[selected_vector][Qi] > max_Qvalue - 0.01 && sol[Qi] != sol[selected_vector]) {
							equ_cnt++;
							if (!(random_engine() % equ_cnt)) {
								max_Qvalue_index = Qi;
							}
						}
					}
					if (max_Qvalue_index != -1) {
						double new_obj_val = move_1(sol, selected_vector, max_Qvalue_index, p_sum);
						// Check for improvement
						if (new_obj_val < best_obj_val - 0.1) {
							best_i = selected_vector;
							best_p = max_Qvalue_index;
							best_obj_val = new_obj_val;
							reward = best_obj_val - new_obj_val;
							impr = 1;
							Q_op++;
						}
					}
				}
			}
			if (impr) {
				maps_count[sol[best_i]]--;
				maps_count[best_p]++;
				int tmp_ori_set = sol[best_i];
				// Update partial sums
				for (int j = 0; j < problem->m; j++) {
					p_sum[sol[best_i]][j] -= weight_mat(best_i, j);
					p_sum[best_p][j] += weight_mat(best_i, j);
				}
				sol[best_i] = best_p;
				obj_val = best_obj_val;
				impr = 1;
				double Q_prime = 0;
				for (int p = 0; p < k; p++) {
					if (p == best_p)
						continue;
					double new_obj_val = move_1(sol, best_i, p, p_sum);
					if (new_obj_val < best_obj_val - 0.1) {
						Q_prime = max(best_obj_val - new_obj_val, Q_prime);
					}
				}
				Q_move[best_i][best_p] = Q_move[best_i][best_p] + para_alpha * (reward + para_gamma * Q_prime - Q_move[best_i][best_p]);
			}
		}
		if (r >= 2) {
			int best_i = -1;
			int best_j = -1;
			double best_obj_val = obj_val;
			// 2-swap operation
			for (int i = random_engine() % problem->n, cnt = 0; cnt < n; cnt++) {
				int selected_vector = (i + cnt) % problem->n;
				if (selected_vector == 0) {
					continue;
				}
				float e_greedy_random_para = e_greedy_random_generator(random_engine);
				if (e_greedy_random_para <= epsilon) {
					for (int j = random_engine() % selected_vector, cnt_j = 0; cnt_j < selected_vector; cnt_j++) {
						int selected_vector_2 = (j + cnt_j) % selected_vector;
						if (sol[selected_vector] == sol[selected_vector_2])
							continue;
						double new_obj_val = swap_2(sol, selected_vector, selected_vector_2, p_sum);
						if (new_obj_val < best_obj_val - 0.1) {
							best_obj_val = new_obj_val;
							best_i = selected_vector;
							best_j = selected_vector_2;
							impr = 1;
							break;
						}
					}
					// According to Q-value matrix
				}
				else {
					int max_Qvalue_index = -1;
					int max_Qvalue = -1;
					int equ_cnt = 1;
					int Qi;
					for (Qi = 0; Qi < problem->n; Qi++) {
						if (Q_swap[selected_vector][Qi] > max_Qvalue && Qi != sol[selected_vector]) {
							max_Qvalue = Q_swap[selected_vector][Qi];
							max_Qvalue_index = Qi;
						}
						else if (Q_swap[selected_vector][Qi] == max_Qvalue && Qi != sol[selected_vector]) {
							equ_cnt++;
							if (!(random_engine() % equ_cnt)) {
								max_Qvalue_index = Qi;
							}
						}
					}
					if (max_Qvalue_index != -1) {
						double new_obj_val = swap_2(sol, selected_vector, max_Qvalue_index, p_sum);
						// Check for improvement
						if (new_obj_val < best_obj_val - 0.1) {
							best_i = selected_vector;
							best_j = max_Qvalue_index;
							best_obj_val = new_obj_val;
							reward = best_obj_val - new_obj_val;
							impr = 1;
							Q_op++;
						}
					}
				}
			}
			if (impr) {
				for (int s = 0; s < problem->m; s++) {
					p_sum[sol[best_i]][s] -= weight_mat(best_i, s);
					p_sum[sol[best_j]][s] -= weight_mat(best_j, s);
					p_sum[sol[best_i]][s] += weight_mat(best_j, s);
					p_sum[sol[best_j]][s] += weight_mat(best_i, s);
				}
				int pi = sol[best_i];
				sol[best_i] = sol[best_j];
				sol[best_j] = pi;
				obj_val = best_obj_val;
				impr = 1;
				double Q_prime = 0;
				for (int i = 0; i < n; i++) {
					if (sol[i] == sol[best_j])
						continue;
					double new_obj_val = swap_2(sol, i, best_j, p_sum);
					if (new_obj_val < best_obj_val - 0.1) {
						Q_prime = max(best_obj_val - new_obj_val, Q_prime);
					}
				}
				Q_swap[best_j][best_i] = Q_swap[best_j][best_i] + para_alpha * (reward + para_gamma * Q_prime - Q_swap[best_j][best_i]);
				Q_prime = 0;
				for (int j = 0; j < n; j++) {
					if (sol[best_i] == sol[j])
						continue;
					double new_obj_val = swap_2(sol, best_i, j, p_sum);
					if (new_obj_val < best_obj_val - 0.1) {
						Q_prime = max(best_obj_val - new_obj_val, Q_prime);
					}
				}
				Q_swap[best_i][best_j] = Q_swap[best_i][best_j] + para_alpha * (reward + para_gamma * Q_prime - Q_swap[best_i][best_j]);
			}
		}
	}
	return obj_val;
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <file_in_name> <n> <m> <k>" << std::endl;
		return 1;
	}

	/*
	 * argv[1] file_num (a, b, c, d, e)
	 * argv[2] n
	 * argv[3] m
	 * argv[4] k
	 */

	problem = new problem_struct;

	string file_num = argv[1];

	string file_in_name = "/home/skkedd/MDMWNPP/mdtwnpp_500_20" + file_num + ".txt";

	problem->n = stoi(argv[2]);
    problem->m = stoi(argv[3]);
    problem->k = stoi(argv[4]);

	std::ifstream in(file_in_name, std::ios::in);
	if (!in.is_open()) {
		std::cout << "open error!" << std::endl;
		exit(0);
	}

	int max_n, max_m;
	in >> max_n >> max_m;

	if (problem->n > max_n || problem->m > max_m) {
		exit(0);
	}

	problem->weight = new double[problem->n * problem->m];
	problem->solution = new double[problem->n];
	problem->sum = new double[problem->m];

	for (int i = 0; !in.eof() && i < problem->n; ++i) {
		for (int j = 0; j < problem->m; ++j) {
			in >> weight_mat(i, j);
		}
	}

	clock_t start_time = clock();
	// Generate random solution
	std::vector<int> s_bsf_global = RandomizedGenerator();

	std::unordered_map<int, int> maps_count;
	for (auto& x : s_bsf_global) {
		if (maps_count.find(x) == maps_count.end())
			maps_count.insert({ x, 1 });
		else
			maps_count[x]++;
	}

	vector<vector<double>> p_sum(problem->k, vector<double>(problem->m, 0.0));

	// Q-value
	vector<vector<int>> Q_move(problem->n, vector<int>(problem->k, 0));
	vector<vector<int>> Q_swap(problem->n, vector<int>(problem->n, 0));
	// Parameters for Q-value
	double epsilon = 1;
	double para_alpha = 0.6;
	double para_gamma = 0.6;

	int r = 1;
	int rmax = 2;

	double obj_val_best = LS_Qvalue(s_bsf_global, maps_count, r, epsilon,
			p_sum, Q_move, Q_swap, para_alpha, para_gamma);
	rearrange(s_bsf_global);

	// Best solution found for the current random initial solution
	// vector<int> s_bsf_local = s_bsf_global;
	// double obj_val_best_local = obj_val_best;

	clock_t curr_time = clock();

	// Parameters for RNS
	int rns_iter = 0;

	int previous_encounter_iter = 0;
	int encounter_timespan = 0;
	int encounter_timespan_threshold = 0;

	int intense_shake_magnitude = 0;
	int intense_shake_magnitude_max = 1;

	double para_beta = 0.7;

	bool update_flag = true;

	HistoryBest history;

	double best_sol_time;

	if (problem->n == 50) {
		encounter_timespan_threshold = 14;
		if(problem->m == 2 || problem->k == 5) {
			intense_shake_magnitude_max = 3;
		}
	}
	else if (problem->n == 100) {
		encounter_timespan_threshold = 10;
	}
	else if (problem->n == 500) {
		encounter_timespan_threshold = 5;
		double para_beta = 1.6;
	}

	double para_m_k = sqrt(sqrt(problem->m) * problem->k);
	double para_n = sqrt(problem->n);

	// Start RLS algorithm
	while ((curr_time - start_time) / CLOCKS_PER_SEC < alg_time_limit) {
		rns_iter++;
		vector<int> s_curr = s_bsf_global;
		// vector<int> s_curr = s_bsf_local;
		// Count the number of elements in each partition
		if (update_flag) {
			maps_count.clear();
			for (auto& x : s_curr) {
				if (maps_count.find(x) == maps_count.end()) {
					maps_count.insert({ x, 1 });
				}
				else {
					maps_count[x]++;
				}
			}
		}

		intense_shake_magnitude = min(intense_shake_magnitude_max, int(sqrt(no_more_impr) * para_n / para_m_k));

		// Shake current solution
		shake(s_curr, maps_count, intense_shake_magnitude);

		// Local search from the current solution 
		double obj_val_curr = LS_Qvalue(s_curr, maps_count, r, epsilon,
			p_sum, Q_move, Q_swap, para_alpha, para_gamma);
		bool already_contain = history.contains(int(obj_val_curr));
		history.insert(obj_val_curr, rns_iter);
		// Improved the global best solution
		if (obj_val_curr < obj_val_best - 0.01) {
			rearrange(s_curr);
			s_bsf_global = s_curr;
			obj_val_best = obj_val_curr;
			previous_encounter_iter = rns_iter;
			r = 1;
			no_more_impr = 0;
			update_flag = true;
			curr_time = clock();
			best_sol_time = (double)(curr_time - start_time) / CLOCKS_PER_SEC;
		}
		else if (already_contain) {
			int encounter_timespan = history.getIterationDifference();
			if (encounter_timespan > encounter_timespan_threshold * para_beta) {
				no_more_impr -= 2;
			}
			r++;
			if (r > rmax)	r = 1;
			no_more_impr++;
			no_more_impr = max(0, no_more_impr);
			update_flag = false;
		}
		else {
			r++;
			if (r > rmax)	r = 1;
			no_more_impr++;
			update_flag = false;
		}
		curr_time = clock();
		if ((curr_time - start_time) / CLOCKS_PER_SEC > alg_time_limit / 3 * 2) {
			epsilon = 0.7;
		}
		else if ((curr_time - start_time) / CLOCKS_PER_SEC > alg_time_limit / 3) {
			epsilon = 0.9;
		}
		if ((curr_time - start_time) / CLOCKS_PER_SEC > alg_time_limit / 100 * statistic_index) {
			statistic_array[statistic_index] = to_string(obj_val_best);
            time_array[statistic_index] = (curr_time - start_time) / CLOCKS_PER_SEC;
            statistic_index++;
		}
	}

	clock_t end_time = clock();
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	obj_val_best = objective(s_bsf_global);

	// save stats to file
	string fname = "/home/skkekk/MDMWNPP/npp_rns_" + file_num + "_result.txt";
	// check if file exists
	ifstream file(fname);
	if (!file.is_open()) {
		// create file
		ofstream myfileOut(fname);
	}
	file.close();
	// write to file
	ofstream outFile(fname, ios::app);
	outFile
		<< file_in_name << ','
		<< to_string(problem->n) << ','
		<< to_string(problem->m) << ','
		<< to_string(problem->k) << ','
		<< to_string(obj_val_best) << ','
		<< to_string(best_sol_time) << ','
		<< to_string(total_time) << ','
		<< endl;

	// for(int i = 1; i < statistic_index; i++) {
	// 	outFile
	// 		<< time_array[i] << ','
	// 		<< statistic_array[i] << ',';
	// }

	// outFile
	// 	<< endl;

	outFile.close();

	return 0;
}