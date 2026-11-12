args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
suppressPackageStartupMessages(library(survey))
d <- read.csv(args[1])
stopifnot(all(d$weight > 0), all(d$domain %in% c(0, 1)))
z <- d[d$domain == 1, ]
stopifnot(nrow(z) == 1341, sum(z$outcome) == 187,
          all(z$p0 > 0 & z$p0 < 1), all(z$p1 > 0 & z$p1 < 1))

# Independent weighted case-control concordance calculation, including ties.
auc_influence <- function(y, score, w) {
  cases <- which(y == 1); controls <- which(y == 0)
  compare <- outer(score[cases], score[controls], function(a,b) (a>b)+0.5*(a==b))
  case_rate <- as.vector(compare %*% w[controls] / sum(w[controls]))
  control_rate <- as.vector(crossprod(w[cases], compare) / sum(w[cases]))
  auc <- sum(w[cases]*case_rate) / sum(w[cases])
  influence <- numeric(length(y))
  influence[cases] <- w[cases] / sum(w[cases]) * (case_rate - auc)
  influence[controls] <- w[controls] / sum(w[controls]) * (control_rate - auc)
  stopifnot(abs(sum(influence)) < 1e-12)
  list(auc=auc, influence=influence)
}
a0 <- auc_influence(z$outcome, z$p0, z$weight)
a1 <- auc_influence(z$outcome, z$p1, z$weight)
for (i in c(1, which(z$outcome==1)[1], which(z$outcome==0)[1], nrow(z))) {
  w <- z$weight; w[i] <- w[i] * (1+1e-6)
  derivative <- (auc_influence(z$outcome,z$p1,w)$auc-a1$auc)/1e-6
  stopifnot(abs(derivative-a1$influence[i]) < 1e-7)
}
d$i0 <- 0; d$i1 <- 0; d$idelta <- 0
d$i0[d$domain==1] <- a0$influence/z$weight
d$i1[d$domain==1] <- a1$influence/z$weight
d$idelta[d$domain==1] <- (a1$influence-a0$influence)/z$weight
d$brier0 <- (d$outcome-d$p0)^2; d$brier1 <- (d$outcome-d$p1)^2
d$logit0 <- qlogis(d$p0); d$logit1 <- qlogis(d$p1)
results <- list()

for (method in c('adjust','average','certainty')) for (domain_adjust in c(TRUE,FALSE)) {
  options(survey.lonely.psu=method, survey.adjust.domain.lonely=domain_adjust)
  des <- svydesign(ids=~psu, strata=~stratum, weights=~weight, data=d, nest=TRUE)
  sub <- subset(des, domain==1)
  df <- degf(sub); critical <- qt(.975,df)
  get_svy_se <- function(expr) suppressWarnings(unname(SE(svytotal(expr,sub))))
  values <- list(
    prevalence=c(unname(coef(suppressWarnings(svymean(~outcome,sub)))), unname(SE(suppressWarnings(svymean(~outcome,sub))))),
    auc_base=c(a0$auc,get_svy_se(~i0)),
    auc_expanded=c(a1$auc,get_svy_se(~i1)),
    auc_difference=c(a1$auc-a0$auc,get_svy_se(~idelta)),
    brier_base=c(unname(coef(suppressWarnings(svymean(~brier0,sub)))),unname(SE(suppressWarnings(svymean(~brier0,sub))))),
    brier_expanded=c(unname(coef(suppressWarnings(svymean(~brier1,sub)))),unname(SE(suppressWarnings(svymean(~brier1,sub)))))
  )
  for (model in c(0,1)) {
    intercept <- suppressWarnings(svyglm(as.formula(paste0('outcome ~ offset(logit',model,')')),
                                         design=sub,family=quasibinomial()))
    slope <- suppressWarnings(svyglm(as.formula(paste0('outcome ~ logit',model)),
                                     design=sub,family=quasibinomial()))
    suffix <- if (model==0) 'base' else 'expanded'
    values[[paste0('citl_',suffix)]] <- c(unname(coef(intercept)[1]),unname(SE(intercept)[1]))
    values[[paste0('calibration_slope_',suffix)]] <- c(unname(coef(slope)[2]),unname(SE(slope)[2]))
  }
  for (metric in names(values)) {
    point <- values[[metric]][1]; se <- values[[metric]][2]
    results[[length(results)+1]] <- data.frame(method=method,domain_adjust=domain_adjust,metric=metric,point=point,se=se,
                                                low=point-critical*se,high=point+critical*se,domain_df=df,
                                                scope='exploratory, fixed models and assumed component design')
  }
}
write.csv(do.call(rbind,results),args[2],row.names=FALSE)
