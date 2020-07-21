data{
    int <lower=1> M; // number of countries
    int <lower=1> P; // number of covariates
    int <lower=1> N0; // number of days for which to impute infections
    int<lower=1> N[M]; // days of observed data for country m. each entry must be <= N2
    int<lower=1> N2; // days of observed data + # of days to forecast
    int cases[N2,M]; // reported cases
    int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
    matrix[N2, M] f; // h * s
    matrix[N2, P] X[M]; // features matrix
    int EpidemicStart[M];
    real pop[M];
    real SI[N2]; // fixed pre-calculated SI using emprical data from Neil
    real masks[N2, M]; // mask features (indicators)
}

transformed data {
    vector[N2] SI_rev; // SI in reverse order
    vector[N2] f_rev[M]; // f in reversed order
  
    for(i in 1:N2)
        SI_rev[i] = SI[N2-i+1];
    
    for(m in 1:M){
        for(i in 1:N2) {
            f_rev[m, i] = f[N2-i+1,m];
        }
    }
}


parameters {
    real<lower=0> mu[M]; // intercept for Rt
    real<lower=0> alpha_hier[8]; // sudo parameter for the hier term for alpha
    real<lower=-0.1> alpha_gaussian[5]; // sudo parameter for Gaussian prior for rollbacks
    real alpha_mask_gaussian[M];         /* how effective are masks, for each region */
    real<lower=0> gamma;
    real<lower=0> kappa;
    real<lower=0> y[M];
    real<lower=0> phi;
    real<lower=0> tau;
    real <lower=0> ifr_noise[M];
}

transformed parameters {
    vector[P] alpha;
    vector[P] masked_alpha;
    real alpha_mask[M];
    matrix[N2, M] prediction = rep_matrix(0,N2,M);
    matrix[N2, M] E_deaths  = rep_matrix(0,N2,M);
    matrix[N2, M] Rt = rep_matrix(0,N2,M);
    matrix[N2, M] Rt_adj = Rt;
    
    {
        matrix[N2,M] cumm_sum = rep_matrix(0,N2,M);
        for(i in 1:8){
            alpha[i] = alpha_hier[i] - ( log(1.05) / 6.0 );
        }
        for(i in 9:13){
            alpha[i] = -alpha_gaussian[i-8];
        }
        for(m in 1:M){
            alpha_mask[m] = alpha_mask_gaussian[m];
        }
        
        for (m in 1:M){
            prediction[1:N0,m] = rep_vector(y[m],N0); // learn the number of cases in the first N0 days
            cumm_sum[2:N0,m] = cumulative_sum(prediction[2:N0,m]);

            /* mask the alphas */
            for (n in 1:N2){
                for (p in 1:P)
                    masked_alpha[p] = alpha[p] * (1 + masks[n,m] * alpha_mask[m]);
                /* masked_alpha[p] = pow(alpha[p], masks[n,m] * alpha_mask[m]); */ // didn't work, because the alpha can go negative, resulting in nan.
                Rt[n,m] = mu[m] * exp(-X[m][n] * masked_alpha);
            }
            
            /* Rt[,m] = mu[m] * exp(-X[m] * alpha); */
            Rt_adj[1:N0,m] = Rt[1:N0,m];
            for (i in (N0+1):N2) {
                real convolution = dot_product(sub_col(prediction, 1, m, i-1), tail(SI_rev, i-1));
                cumm_sum[i,m] = cumm_sum[i-1,m] + prediction[i-1,m];
                Rt_adj[i,m] = ((pop[m]-cumm_sum[i,m]) / pop[m]) * Rt[i,m];
                prediction[i, m] = Rt_adj[i,m] * convolution;
            }
            E_deaths[1, m]= 1e-15 * prediction[1,m];
            for (i in 2:N2){
                E_deaths[i,m] = ifr_noise[m] * dot_product(sub_col(prediction, 1, m, i-1), tail(f_rev[m], i-1));
            }
        }
    }
}
model {
    tau ~ exponential(0.03);
    for (m in 1:M){
        y[m] ~ exponential(1/tau);
    }
    phi ~ normal(0,5);
    kappa ~ normal(0,0.5);
    mu ~ normal(3.28, kappa); // citation: https://academic.oup.com/jtm/article/27/2/taaa021/5735319
    alpha_hier ~ gamma(.1667,1);
    alpha_gaussian ~ normal(0,0.1);
    alpha_mask_gaussian ~ normal(0,0.1);    
    ifr_noise ~ normal(1,0.1);
    for(m in 1:M){
        deaths[EpidemicStart[m]:N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m]:N[m], m], phi);
    }
}


