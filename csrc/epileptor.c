#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define vars_i vars[i]
#define vars_j vars[j]
#define PI 3.14159265358979

/**
 * parameters of 5d epileptor model
 * 
 * References:
 *     Jirsa, Viktor K., William C. Stacey, Pascale P. Quilichini, Anton I. Ivanov和Christophe Bernard. 2014. 
 *     《On the Nature of Seizure Dynamics》. Brain 137 (8): 2210–30. https://doi.org/10.1093/brain/awu133.
 */
struct parameter_epileptor_5 {
    double x_0;
    double y_0;
    double tau_0;
    double tau_1;
    double tau_2;
    double I_rest1;
    double I_rest2;
    double gamma;
    double D_1;
    double D_2;
};

/**
 * variables of 5d epileptor model
 * 
 * References:
 *     Jirsa, Viktor K., William C. Stacey, Pascale P. Quilichini, Anton I. Ivanov和Christophe Bernard. 2014. 
 *     《On the Nature of Seizure Dynamics》. Brain 137 (8): 2210–30. https://doi.org/10.1093/brain/awu133.
 */
struct epileptor_5 {
    double x_1;
    double y_1;
    double z;
    double x_2;
    double y_2;
    double u;
    struct parameter_epileptor_5 *paras;
};

struct epileptor_network {
    struct epileptor_5 **vars;
    double **K;
    int n;
    int **Tau;
};

struct epileptor_result {
    double **x_1s;
    double **zs;
    double **x_2s;
    double **Izs;
};

struct epileptor_result_full {
    double **x_1s;
    double **y_1s;
    double **zs;
    double **x_2s;
    double **y_2s;
    double **Izs;
};

void init_rand() {
    srand((int) time(0));
}

double uniform_rand(double l, double u, unsigned int *seed) {
    return l + ((double) rand_r(seed) / (RAND_MAX + 1.0)) * (u - l);
}

double gauss_rand(double mu, double sigma, unsigned int *seed) {
    double u1,u2;
    do {
        u1 = uniform_rand(0, 1, seed);
        u2 = uniform_rand(0, 1, seed);
    } while (u1 <= 0);
    double z0  = sqrt(- 2.0 * log(u1)) * cos(2 * PI * u2);
    return sigma * z0 + mu;
}

void odes_epileptor_5(struct epileptor_5 **vars, double **K, double *Iz, int n, double dt, unsigned int *seed) {
    double x_1 = 0;
    double y_1 = 0;
    double   z = 0;
    double x_2 = 0;
    double y_2 = 0;
    double   u = 0;
    double dx_1 = 0;
    double dy_1 = 0;
    double   dz = 0;
    double dx_2 = 0;
    double dy_2 = 0;
    double   du = 0;
    double  f_1 = 0;
    double  f_2 = 0;
    struct parameter_epileptor_5 *paras;
    for (int i = 0; i < n; i ++) {
        x_1 = vars_i->x_1;
        y_1 = vars_i->y_1;
          z = vars_i->z;
         x_2 = vars_i->x_2;
         y_2 = vars_i->y_2;
           u = vars_i->u;
        paras= vars_i->paras;
        if (x_1 < 0) {
            f_1 = x_1 * x_1 * x_1 - 3 * x_1 * x_1;
        } else {
            f_1 = (x_2 - 0.6 * pow(z - 4, 2)) * x_1;
        }
        dx_1 = y_1 - f_1 - z + paras->I_rest1;
        dy_1 = paras->y_0 - 5 * x_1 * x_1 - y_1;
          dz = (4 * (x_1 - paras->x_0) - z + Iz[i]) / paras->tau_0;
        dx_2 = -y_2 + x_2 - x_2 * x_2 * x_2 + paras->I_rest2 + 2 * u - 0.3 * (z - 3.5);
        if (x_2 < -0.25) {
            f_2 = 0;
        } else {
            f_2 = 6 * (x_2 + 0.25);
        }
        dy_2 = (- y_2 + f_2) / paras->tau_2;
          du = - paras->gamma * (u - 0.1 * x_1);
        vars_i->x_1 += (dx_1 + gauss_rand(0, paras->D_1, seed)) * dt;
        vars_i->y_1 += (dy_1 + gauss_rand(0, paras->D_1, seed)) * dt;
        vars_i->  z += dz   * dt;
        vars_i->x_2 += (dx_2 + gauss_rand(0, paras->D_2, seed)) * dt;
        vars_i->y_2 += (dy_2 + gauss_rand(0, paras->D_2, seed)) * dt;
        vars_i->  u += du   * dt;
    }
}

struct parameter_epileptor_5 *gen_parameter_epileptor_5() {
    struct parameter_epileptor_5 *paras = malloc(sizeof(struct parameter_epileptor_5));
    paras->x_0 = -1.6;
    paras->y_0 = 1;
    paras->tau_0 = 2857;
    paras->tau_1 = 1;
    paras->tau_2 = 10;
    paras->I_rest1 = 3.1;
    paras->I_rest2 = 0.45;
    paras->gamma = 0.01;
    paras->D_1 = 0.025;
    paras->D_2 = 0.25;
    return paras;
}

void init_epileptor_5(struct epileptor_5 *var, unsigned int *seed) {
    var->x_1 = uniform_rand(-1.86, -1.4624, seed);
    var->y_1 = uniform_rand(-15.9443, -9.6935, seed);
    var->z   = uniform_rand(2.9503, 3.58, seed);
    var->x_2 = uniform_rand(-0.954, -0.7581, seed);
    var->y_2 = 0;
    var->u   = 0;
    var->paras=gen_parameter_epileptor_5();
}

struct epileptor_5 **gen_model_vars(int n, unsigned int *seed) {
    struct epileptor_5 **vars = (struct epileptor_5 **) malloc(sizeof(struct epileptor_5*) * n);
    for (int i = 0; i < n; i ++) {
        vars_i = malloc(sizeof(struct epileptor_5));
        init_epileptor_5(vars_i, seed);
    }
    return vars;
}

void init_stable_epileptor_5(struct epileptor_5 *var) {
    var->x_1 = -1.5361;
    var->y_1 = -10.803;
    var->z   = 3;
    var->x_2 = -0.8;
    var->y_2 = 0;
    var->u   = 0;
    var->paras=gen_parameter_epileptor_5();
}

struct epileptor_5 **gen_stable_model_vars(int n, unsigned int *seed) {
    struct epileptor_5 **vars = (struct epileptor_5 **) malloc(sizeof(struct epileptor_5*) * n);
    for (int i = 0; i < n; i ++) {
        vars_i = malloc(sizeof(struct epileptor_5));
        init_stable_epileptor_5(vars_i);
    }
    return vars;
}

struct epileptor_result *init_result(int n, int steps) {
    double **x_1s = (double **) malloc(sizeof(double *) * n);
    double **zs = (double **) malloc(sizeof(double *) * n);
    double **x_2s = (double **) malloc(sizeof(double *) * n);
    double **Izs = (double **) malloc(sizeof(double *) * n);
    for (int i = 0; i < n; i ++) {
        x_1s[i] = malloc(sizeof(double) * steps);
        zs[i] = malloc(sizeof(double) * steps);
        x_2s[i] = malloc(sizeof(double) * steps);
        Izs[i] = malloc(sizeof(double) * steps);
    }
    struct epileptor_result *result = (struct epileptor_result *) malloc(sizeof(struct epileptor_result));
    result->x_1s = x_1s;
    result->zs   = zs;
    result->x_2s = x_2s;
    result->Izs  = Izs;
    return result;
}

struct epileptor_result_full *init_full_result(int n, int steps) {
    double **x_1s = (double **) malloc(sizeof(double *) * n);
    double **y_1s = (double **) malloc(sizeof(double *) * n);
    double **zs = (double **) malloc(sizeof(double *) * n);
    double **x_2s = (double **) malloc(sizeof(double *) * n);
    double **y_2s = (double **) malloc(sizeof(double *) * n);
    double **Izs = (double **) malloc(sizeof(double *) * n);
    for (int i = 0; i < n; i ++) {
        x_1s[i] = malloc(sizeof(double) * steps);
        y_1s[i] = malloc(sizeof(double) * steps);
        zs[i] = malloc(sizeof(double) * steps);
        x_2s[i] = malloc(sizeof(double) * steps);
        y_2s[i] = malloc(sizeof(double) * steps);
        Izs[i] = malloc(sizeof(double) * steps);
    }
    struct epileptor_result_full *result = (struct epileptor_result_full *) malloc(sizeof(struct epileptor_result_full));
    result->x_1s = x_1s;
    result->y_1s = y_1s;
    result->zs   = zs;
    result->x_2s = x_2s;
    result->y_2s = y_2s;
    result->Izs  = Izs;
    return result;
}

struct epileptor_result simulation(struct epileptor_network *network, double dt, unsigned int *seed, int steps) {
    struct epileptor_result *result = init_result(network->n, steps);
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        for (int m = 0; m < network->n; m ++) {
            Iz[m] = 0;
            for (int j = 0; j < network->n; j ++) {
                Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - network->vars[j]->x_1);
            }
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->Izs[j][i]  = Iz[j];
        }
    }
    return *result;
}

struct epileptor_result simulation_extrenal_input(struct epileptor_network *network, double dt, double (*intput)(int, double), unsigned int *seed, int steps) {
    struct epileptor_result *result = init_result(network->n, steps);
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        for (int m = 0; m < network->n; m ++) {
            Iz[m] = 0;
            for (int j = 0; j < network->n; j ++) {
                Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - network->vars[j]->x_1);
            }
            Iz[m] += intput(m, i * dt);
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->Izs[j][i]  = Iz[j];
        }
    }
    return *result;
}

struct epileptor_result_full simulation_full(struct epileptor_network *network, double dt, unsigned int *seed, int steps) {
    struct epileptor_result_full *result = init_full_result(network->n, steps);
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        for (int m = 0; m < network->n; m ++) {
            Iz[m] = 0;
            for (int j = 0; j < network->n; j ++) {
                Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - network->vars[j]->x_1);
            }
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->y_1s[j][i] = network->vars[j]->y_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->y_2s[j][i] = network->vars[j]->y_2;
            result->Izs[j][i]  = Iz[j];
        }
    }
    return *result;
}

struct epileptor_result simulation_delay(struct epileptor_network *network, int max_tau, double dt, unsigned int *seed, int steps) {
    struct epileptor_result *result = init_result(network->n, steps);
    double **xs = (double **) malloc(sizeof(double *) * network->n);
    for (int i = 0; i < network->n; i ++) {
        xs[i] = (double *) malloc(sizeof(double) * (max_tau + 1));
        for (int j = 0; j < max_tau + 1; j ++) {
            xs[i][j] = 0;
        }
    }
    int pt = 0;
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        if (i > max_tau) {
            for (int m = 0; m < network->n; m ++) {
                Iz[m] = 0;
                for (int j = 0; j < network->n; j ++) {
                    int tau_j_m = network->Tau[j][m];
                    double x_j = xs[j][(pt + max_tau - tau_j_m) % (max_tau + 1)];
                    Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - x_j);
                }
            }
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->Izs[j][i]  = Iz[j];
            xs[j][pt] = network->vars[j]->x_1;
        }
        pt ++;
        if (pt > max_tau)
            pt = 0;
    }
    // free(Iz);
    // for (int i = 0; i < network->n; i ++) {
    //     free(xs[i]);
    // }
    // free(xs);
    return *result;
}

struct epileptor_result simulation_sin(struct epileptor_network *network, double dt, unsigned int *seed, int steps, double *phi, double amp, double omega) {
    struct epileptor_result *result = init_result(network->n, steps);
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        double t = i / 256.0;
        for (int m = 0; m < network->n; m ++) {
            Iz[m] = 0;
            for (int j = 0; j < network->n; j ++) {
                Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - network->vars[j]->x_1);
            }
            // add sin fluctuations
            Iz[m] += amp * sin(omega * t + phi[m]);
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->Izs[j][i]  = Iz[j];
        }
    }
    return *result;
}

struct epileptor_result simulation_sin_x1(struct epileptor_network *network, double dt, unsigned int *seed, int steps, double *phi, double amp, double omega) {
    struct epileptor_result *result = init_result(network->n, steps);
    double *Iz = (double *) malloc(sizeof(double) * network->n);
    for (int i = 0; i < steps; i ++) {
        double t = i / 256.0;
        for (int m = 0; m < network->n; m ++) {
            Iz[m] = 0;
            for (int j = 0; j < network->n; j ++) {
                Iz[m] += network->K[j][m] * (network->vars[m]->x_1 - network->vars[j]->x_1);
            }
            // add sin fluctuations
            network->vars[m]->x_1 += amp * sin(omega * t + phi[m]);
        }
        odes_epileptor_5(network->vars, network->K, Iz, network->n , dt, seed);
        for (int j = 0; j < network->n; j ++) {
            result->x_1s[j][i] = network->vars[j]->x_1;
            result->zs[j][i]   = network->vars[j]->z;
            result->x_2s[j][i] = network->vars[j]->x_2;
            result->Izs[j][i]  = Iz[j];
        }
    }
    return *result;
}

void free_network(struct epileptor_network network) {
    for (int i = 0; i < network.n; i ++) {
        free(network.vars[i]);
        // free(network.K[i]);
    }
    free(network.vars);
    // free(network.K);
    // free(network);
}

void free_data(struct epileptor_result result, struct epileptor_network network) {
    for (int i = 0; i < network.n; i ++) {
        free(result.x_1s[i]);
        free(result.zs[i]);
        free(result.x_2s[i]);
        free(result.Izs[i]);
    }
    free(result.x_1s);
    free(result.zs);
    free(result.x_2s);
    free(result.Izs);
    // free(result);
    free_network(network);
}

void free_full_data(struct epileptor_result_full result, struct epileptor_network network) {
    for (int i = 0; i < network.n; i ++) {
        free(result.x_1s[i]);
        free(result.y_1s[i]);
        free(result.zs[i]);
        free(result.x_2s[i]);
        free(result.y_2s[i]);
        free(result.Izs[i]);
    }
    free(result.x_1s);
    free(result.y_1s);
    free(result.zs);
    free(result.x_2s);
    free(result.y_2s);
    free(result.Izs);
    // free(result);
    free_network(network);
}

// result preprocess
int *extract_seizure_event(double *x_1s, int n, int *tpc) {
    int *tps = malloc(sizeof(int));
    int len = 0;
    int state = 0;
    for (int i = 0; i < n; i ++) {
        if (! state && x_1s[i] > 0) {
            state = 1;
            tps[len ++] = i;
            tps = realloc(tps, (len + 1) * sizeof(int));
        } else if (state && x_1s[i] < -1.5) {
            state = 0;
            tps[len ++] = i;
            tps = realloc(tps, (len + 1) * sizeof(int));
        }
    }
    *tpc = len;
    return tps;
}

int main() {
    // #pragma omp parallel for
    // for (int k = 0; k < 8; k ++) {
        unsigned int seed = time(0) + omp_get_thread_num();
        int n = 96;
        struct epileptor_5 **vars = (struct epileptor_5 **) malloc(sizeof(struct epileptor_5*) * n);
        double **K = (double **) malloc(sizeof(double *) * n);
        for (int i = 0; i < n; i ++) {
            vars_i = malloc(sizeof(struct epileptor_5));
            init_epileptor_5(vars_i, &seed);
            vars_i->paras->x_0 = -2.2;
            K[i] = malloc(sizeof(double) * n);
            for (int j = 0; j < n; j ++) {
                K[i][j] = gauss_rand(0, 0.1, &seed);
            }
        }
        vars[0]->paras->x_0 = -1.88;
        int t = 3000 / 0.05;
        double *Iz = (double *) malloc(sizeof(double) * n);
        for (int i = 0; i < t; i ++) {
            for (int i = 0; i < n; i ++) {
                Iz[i] = 0;
                for (int j = 0; j < n; j ++) {
                    Iz[i] += K[j][i] * (vars_i->x_1 - vars_j->x_1);
                }
            }
            odes_epileptor_5(vars, K, Iz, n, 0.05, &seed);
            // for (int j = 0; j < n; j ++) {
            //     printf("%.8lf ", vars_j->x_2 - vars_j->x_1);
            // }
            // printf("\n");
        }
        for (int i = 0; i < n; i ++) {
            free(vars_i);
            free(K[i]);
        }
        free(vars);
        free(K);
        free(Iz);
    // }
    return 0;
}