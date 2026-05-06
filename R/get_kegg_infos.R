

######get KEGG pathways and corresponding metabolic
library(KEGGREST)
library(tcltk)
file.name="QCP11"
output.path="output"
ppm=15



hsa_pathways=keggList("pathway","hsa")
pathway_id=data.frame(names(hsa_pathways))
pathway_compound=list()
for (i in 1:length(pathway_id$names.hsa_pathways)) {
  pathway_compound[[i]]=NA
}
kegg_result=lapply(1:361,function(i){
  print(i)
  x=pathway_id$names.hsa_pathways[i]
    tryCatch({
      temp=keggGet(x)
      if(!is.null(temp[[1]]$COMPOUND)){
        return(data.frame(names(temp[[1]]$COMPOUND)))
      }else{
        return(data.frame(NA))
      }
    },error=function(e){
      print(i)
    })
  })



save.image(file="kegg_pathway_info.Rda")




######get KEGG compounds exact mass
load("kegg_pathway_info.Rda")
index=1
remind_pathway_index=c()
for (i in pathway_compound) {
  if(nrow(i)>1){
    remind_pathway_index=c(remind_pathway_index,index)
  }
  index=index+1
}
kegg_pathways_id=pathway_id[remind_pathway_index,]
kegg_pathways=list()
index=1
for (i in remind_pathway_index) {
  kegg_pathways[[index]]=pathway_compound[[i]]
  index=index+1
}


detail_list=c()
for (x in kegg_pathways) {
  for (j in x$names.temp..1...COMPOUND.) {
    detail_list=c(detail_list,j)
  }
}
detail_list=unique(detail_list)
compounds_detail=data.frame(id=detail_list,exact_mass=0)
index=1
while (index <= nrow(compounds_detail)) {
  print(index)
  temp=KEGGREST::keggGet(detail_list[index])[[1]]
  tryCatch({
    compounds_detail$exact_mass[index]=temp$EXACT_MASS
  },error=function(e){
    
  })
  index=index+1
}


#nrow(compounds_detail[compounds_detail$exact_mass==0,])
compounds_detail_res=compounds_detail[!compounds_detail$exact_mass==0,]
compounds_detail_res$exact_mass=as.numeric(compounds_detail_res$exact_mass)
compounds_detail_res$mz_min=0
for (i in 1:nrow(compounds_detail_res)) {
  compounds_detail_res$mz_min[i]=ifelse(compounds_detail_res$exact_mass[i]>400,
                                        compounds_detail_res$exact_mass[i]-ppm*compounds_detail_res$exact_mass[i]/100000,
                                        compounds_detail_res$exact_mass[i]-0.0004*ppm)
  compounds_detail_res$mz_max[i]=ifelse(compounds_detail_res$exact_mass[i]>400,
                                        compounds_detail_res$exact_mass[i]+ppm*compounds_detail_res$exact_mass[i]/100000,
                                        compounds_detail_res$exact_mass[i]+0.0004*ppm)
}
rownames(compounds_detail_res)=compounds_detail_res$id
save.image(file="kegg_pathway_info_detail.Rda")
save(compounds_detail_res,file="compounds_detail_res.Rda")
save(kegg_pathways,file="kegg_pathways.Rda")
save(kegg_pathways_id,file="kegg_pathways_id.Rda")
rm(list=ls())

load(file="compounds_detail_res.Rda")
load(file="kegg_pathways.Rda")
load(file="kegg_pathways_id.Rda")

reactions=keggList("Reaction")
ping=function(x,t){
  if(t>10){
    print(x)
    return(NULL)
  }
  tryCatch({
    return(keggGet(rownames(data.frame(reactions[x])))[[1]])
  },error=function(e){
    Sys.sleep(5)
    return(ping(x,t+1))
  })
}
pb <- txtProgressBar(style=3)
search_kegg_react=function(x){
  setTxtProgressBar(pb, x/length(reactions))
  request=ping(x,1)
  enzy=request$ENZYME
  temp=request$EQUATION
  #temp=strsplit(reactions[x],"; ")[[1]]
  #temp=temp[length(temp)]
  temp=strsplit(temp," <=> ")[[1]]
  rec=temp[1]
  prod=temp[2]
  rec=strsplit(rec," \\+ ")[[1]]
  prod=strsplit(prod," \\+ ")[[1]]
  rec=data.frame(metbolite=sub("^\\s*\\S+\\s+", "", rec),source="rec")
  prod=data.frame(metbolite=sub("^\\s*\\S+\\s+", "", prod),source="prod")
  if(!is.null(enzy)){
    enzy=data.frame(metbolite=enzy,source="enzy")
  }
  return(rbind(rec,prod,enzy))
}
formulars=lapply(1:length(reactions),search_kegg_react)

links=do.call(rbind,lapply(1:length(formulars),function(x){
  temp=formulars[[x]]
  rec=temp[temp$source=='rec',]
  prod=temp[temp$source=='prod',]
  enzy=temp[temp$source=='enzy',]
  ret=data.frame()
  for (i in 1:nrow(rec)) {
    if(nrow(enzy)==0){
      for (j in 1:nrow(prod)) {
        ret=rbind(ret,data.frame(from=rec$metbolite[i],to=prod$metbolite[j]))
      }
    }else{
      for (j in 1:nrow(enzy)) {
        for(k in 1:nrow(prod)){
          ret=rbind(ret,data.frame(from=rec$metbolite[i],to=enzy$metbolite[j]))
          ret=rbind(ret,data.frame(from=enzy$metbolite[j],to=prod$metbolite[k]))
        }
      }
    }
    
  }
  return(ret)
}))
save(formulars,file="formulars.Rda")
save(links,file="links.Rda")
save(reactions,file="reactions.Rda")

openxlsx::write.xlsx(links,"links.xlsx")































