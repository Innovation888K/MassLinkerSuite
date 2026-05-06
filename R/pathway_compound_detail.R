tp=data.frame()
rownames(compounds_detail_res)=compounds_detail_res$id
for (i in 1:length(kegg_pathways)) {
  t=kegg_pathways[[i]]
  t=t[,1][t[,1]%in%compounds_detail_res$id]
  if(length(t)>0){
    tp=rbind(tp,data.frame(kegg_id=unlist(kegg_pathways_id[i]),
                            pathway_name=unlist(hsa_pathways[kegg_pathways_id[i]]),
                            compound_names=t,
                           mz=compounds_detail_res[t,2]))  
  }
}
write.csv(tp,file="pathway_compound_detail.csv")
