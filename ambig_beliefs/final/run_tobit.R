library(jsonlite)
library(arrow)
library(AER)
library(marginaleffects)
library(DescTools)
library(multcomp)

perform_wald_tests <- function(model, info, n_types, cluster_vcov) {
    for (j in 2:(n_types - 1)) {
        for (i in 1:(j - 1)) {

            # Build up test matrix
            test_spec <- matrix(0, 1, length(coef(model)))
            colnames(test_spec) <- names(coef(model))
            test_spec[1, i + 1] <- 1
            test_spec[1, j + 1] <- -1

            # Perform test and save in info object
            wald_test = glht(model, linfct = test_spec)
            sum_wald_test = summary(wald_test, vcov = cluster_vcov)
            key = paste(c("Type", as.character(i), "-", "Type", as.character(j)), collapse = "")
            info[[key]] = sum_wald_test$test$pvalues[1]
        }
    }
    return(info)
}

save_marginal_effects_and_infos_probit <- function(model, n_types, perform_wald, cluster_vcov, path_out, path_out_info) {
    # print(path_out)
    # print(path_out_info)

    marg_eff = summary(marginaleffects(model, vcov = cluster_vcov))
    write_parquet(marg_eff, path_out)

    # save infos of regression
    obs = length(model$fitted.values)
    pseudo_r2 = PseudoR2(model)
    info = list("n_obs"=obs, "pseudo_r2"=pseudo_r2)

    # Perform Wald-tests
    if (perform_wald) {
        info = perform_wald_tests(model, info, n_types, cluster_vcov)
    }
    write_json(info, path_out_info)
}

save_marginal_effects_and_infos_tobit <- function(model, n_types, perform_wald, cluster_vcov, path_out, path_out_info) {

    marg_eff = summary(marginaleffects(model, vcov = cluster_vcov))
    write_parquet(marg_eff, path_out)

    # save infos of regression
    summary = summary(model)
    obs = summary(model)$n["Total"]
    pseudo_r2 = 1 - summary(model)$loglik[2] / summary(model)$loglik[1]
    info = list("n_obs"=obs, "pseudo_r2"=pseudo_r2)

    # Perform Wald-tests
    if (perform_wald) {
        info = perform_wald_tests(model, info, n_types, cluster_vcov)
    }
    write_json(info, path_out_info)
}

# Read pytask spec files
args <- commandArgs(trailingOnly=TRUE)
path_to_json <- args[length(args)]
config <- read_json(path_to_json)


# Load file
df <- read_parquet(file = config$depends_on$reg_sample_r)
# df <- na.omit(df)
# df <- read_parquet(file = 'C:/ECON/ambig-beliefs/ambig_beliefs/sandbox/data_for_tobit.parquet')
# df$type_man_sort <- df$k4

df$type_man_sort <- as.factor(df$type_man_sort)
df$age_groups <- as.factor(df$age_groups)
df$net_income_groups <- as.factor(df$net_income_groups)
df$total_financial_assets_groups <- as.factor(df$total_financial_assets_groups)
df$female <- as.logical(df$female)

n_types = length(unique(df$type_man_sort))

if (config$cluster_std) {
    cluster_vcov <- as.formula("~personal_id")
} else {
    cluster_vcov <- "HC3"

}

# Probit: short
probit_AER <- glm(as.formula(paste("has_rfa ~ ", config$formula_short)), family=binomial(link = "probit"), data = df)
save_marginal_effects_and_infos_probit(probit_AER, n_types, config$ambig_types, cluster_vcov, config$produces$results_probit_short, config$produces$results_info_probit_short)

# Probit: controls
probit_AER <- glm(as.formula(paste("has_rfa ~ ", config$formula_controls)), family=binomial(link = "probit"), data = df)
save_marginal_effects_and_infos_probit(probit_AER, n_types, config$ambig_types, cluster_vcov, config$produces$results_probit_controls, config$produces$results_info_probit_controls)

# Tobit: short
tobit_AER <- tobit(as.formula(paste("frac_of_tfa_in_rfa ~ ", config$formula_short)), left=0, data = df)
save_marginal_effects_and_infos_tobit(tobit_AER, n_types, config$ambig_types, cluster_vcov, config$produces$results_tobit_short, config$produces$results_info_tobit_short)

# Tobit: controls
tobit_AER <- tobit(as.formula(paste("frac_of_tfa_in_rfa ~ ", config$formula_controls)), left=0, data = df)
save_marginal_effects_and_infos_tobit(tobit_AER, n_types, config$ambig_types, cluster_vcov, config$produces$results_tobit_controls, config$produces$results_info_tobit_controls)

# # Run regressions again for subsets of the data
# if (!config$ambig_types) {
#     for (sel_var in c("all_indices_valid", "at_least_2_waves_with_valid_choice_and_index")) {
#         df_sel = df[df[, sel_var] == TRUE, ]

#         # Probit: short
#         probit_AER <- glm(as.formula(paste("has_rfa ~ ", config$formula_short)), family=binomial(link = "probit"), data = df_sel)
#         save_marginal_effects_and_infos_probit(probit_AER, n_types, config$ambig_types, cluster_vcov, config$produces[[paste("results_probit_short", sel_var, sep="_")]], config$produces[[paste("results_info_probit_short", sel_var, sep="_")]])

#         # Probit: controls
#         probit_AER <- glm(as.formula(paste("has_rfa ~ ", config$formula_controls)), family=binomial(link = "probit"), data = df_sel)
#         save_marginal_effects_and_infos_probit(probit_AER, n_types, config$ambig_types, cluster_vcov, config$produces[[paste("results_probit_controls", sel_var, sep="_")]], config$produces[[paste("results_info_probit_controls", sel_var, sep="_")]])

#         # Tobit: short
#         tobit_AER <- tobit(as.formula(paste("frac_of_tfa_in_rfa ~ ", config$formula_short)), left=0, data = df_sel)
#         save_marginal_effects_and_infos_tobit(tobit_AER, n_types, config$ambig_types, cluster_vcov, config$produces[[paste("results_tobit_short", sel_var, sep="_")]], config$produces[[paste("results_info_tobit_short", sel_var, sep="_")]])

#         # Tobit: controls
#         tobit_AER <- tobit(as.formula(paste("frac_of_tfa_in_rfa ~ ", config$formula_controls)), left=0, data = df_sel)
#         save_marginal_effects_and_infos_tobit(tobit_AER, n_types, config$ambig_types, cluster_vcov, config$produces[[paste("results_tobit_controls", sel_var, sep="_")]], config$produces[[paste("results_info_tobit_controls", sel_var, sep="_")]])
#     }
# }
# marg_eff <- function(df, formula){
#     library(censReg)

#     tobit <- censReg(formula, left=0, right=1, data = df)

#     return(summary(margEff(tobit)))
# }

# info <- function(df, formula){
#     library(censReg)
#     tobit <- censReg(formula, left=0, right=1, data = df)

#     # calculate pseudoR2 and build named vector of information
#     pseudoR2 <- function(obj) 1 - as.vector(logLik(obj)/logLik(update(obj, . ~ 1)))
#     res <- data.frame("nObs"=c(summary(tobit)$nObs["Total"]),"pseudo_R2"=c(pseudoR2(tobit)))
#     return(res)
# }