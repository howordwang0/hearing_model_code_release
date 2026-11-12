args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
suppressPackageStartupMessages(library(survey))
d <- read.csv(args[1])
stopifnot(all(d$weight > 0), all(d$domain %in% c(0, 1)))
rows <- list()
for (method in c('adjust', 'average', 'certainty')) {
  options(survey.lonely.psu = method, survey.adjust.domain.lonely = TRUE)
  design <- svydesign(ids = ~psu, strata = ~stratum, weights = ~weight,
                      data = d, nest = TRUE)
  domain <- subset(design, domain == 1)
  value <- withCallingHandlers({
    a <- svytotal(~influence_per_weight, domain)
    prev <- svymean(~outcome, domain)
    degrees <- degf(domain)
    data.frame(method = method, delta_se = unname(SE(a)), domain_df = degrees,
               prevalence = unname(coef(prev)), prevalence_se = unname(SE(prev)),
               status = 'exploratory design assumption')
  }, warning = function(w) {
    message('method=', method, ': ', conditionMessage(w))
    invokeRestart('muffleWarning')
  })
  rows[[length(rows)+1]] <- value
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
