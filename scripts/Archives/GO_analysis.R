if (!requireNamespace ("BiocManager" , quietly = TRUE )) install.packages("BiocManager")
  BiocManager :: install(c("beadarray", "limma", "GEOquery", "illuminaHumanv1.db", "illuminaHumanv2.db" , "illuminaHumanv3.db", "illuminaHumanv4.db",
                      "BeadArrayUseCases", "GOfuncR", 'Pigengene', 'biomaRt'))


library('annotate')
library("illuminaHumanv1.db")
x = getGO("GI_10047089-S","illuminaHumanv1")
x = x$`GI_10047089-S`

sapply(x, function(x) x$Ontology)

getOntology(x, "CC")
zz = dropECode(x)
getEvidence(zz) # drop IEA


BPisa = eapply(GOBPPARENTS, function(x) names(x))
table(unlist(BPisa))
#is_a negatively_regulates part_of
#57509 2813 5388
#positively_regulates regulates
#2796 3240
MFisa = eapply(GOMFPARENTS, function(x) names(x))
table(unlist(MFisa))
#is_a part_of
#13671 11
CCisa = eapply(GOCCPARENTS, function(x) names(x))
table(unlist(CCisa))



res_hyper = go_enrich(input_hyper, n_randset=100)

'GI_10047089-S'
'GI_10047091-S'
'GI_10047099-S'
