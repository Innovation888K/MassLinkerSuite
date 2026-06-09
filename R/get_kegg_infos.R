# Retrieve KEGG pathway, compound, and reaction metadata for MassLinker.
#
# This script downloads KEGG human pathway information, extracts pathway-linked
# compounds, retrieves compound exact masses, calculates m/z matching windows,
# and constructs reaction-derived links between metabolites and enzymes.
#
# Generated files include:
#   - kegg_pathway_info.Rda
#   - kegg_pathway_info_detail.Rda
#   - compounds_detail_res.Rda
#   - kegg_pathways.Rda
#   - kegg_pathways_id.Rda
#   - formulars.Rda
#   - links.Rda
#   - reactions.Rda
#   - links.xlsx
#
# Notes:
#   This script queries KEGG online services through KEGGREST, so execution time
#   and success may depend on network stability and KEGG service availability.


###### Get KEGG pathways and corresponding metabolites.
library(KEGGREST)
library(tcltk)

file.name="QCP11"
output.path="output"
ppm=15


# Retrieve all human KEGG pathways.
hsa_pathways=keggList("pathway","hsa")

pathway_id=data.frame(names(hsa_pathways))

pathway_compound=list()

for (i in 1:length(pathway_id$names.hsa_pathways)) {
  pathway_compound[[i]]=NA
}

# Query compounds associated with each KEGG pathway.
kegg_result=lapply(1:nrow(pathway_id),function(i){
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


# Save the current workspace containing pathway query results.
save.image(file="kegg_pathway_info.Rda")




###### Get KEGG compound exact masses.
load("kegg_pathway_info.Rda")

index=1
remind_pathway_index=c()

# Keep pathways that contain more than one associated compound entry.
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


# Collect unique KEGG compound identifiers from retained pathways.
detail_list=c()

for (x in kegg_pathways) {
  for (j in x$names.temp..1...COMPOUND.) {
    detail_list=c(detail_list,j)
  }
}

detail_list=unique(detail_list)

compounds_detail=data.frame(id=detail_list,exact_mass=0)

index=1

# Retrieve exact masses for all collected KEGG compounds.
while (index <= nrow(compounds_detail)) {
  print(index)

  temp=KEGGREST::keggGet(detail_list[index])[[1]]

  tryCatch({
    compounds_detail$exact_mass[index]=temp$EXACT_MASS
  },error=function(e){
    
  })

  index=index+1
}


# Count compounds without exact mass if needed.
# nrow(compounds_detail[compounds_detail$exact_mass==0,])

# Retain compounds with valid exact mass annotations.
compounds_detail_res=compounds_detail[!compounds_detail$exact_mass==0,]

compounds_detail_res$exact_mass=as.numeric(compounds_detail_res$exact_mass)

compounds_detail_res$mz_min=0

# Calculate compound-specific m/z matching windows.
# For masses above 400 Da, use ppm scaling. For smaller masses, use a fixed
# 0.0004 * ppm absolute tolerance.
for (i in 1:nrow(compounds_detail_res)) {
  compounds_detail_res$mz_min[i]=ifelse(compounds_detail_res$exact_mass[i]>400,
                                        compounds_detail_res$exact_mass[i]-ppm*compounds_detail_res$exact_mass[i]/1000000,
                                        compounds_detail_res$exact_mass[i]-0.0004*ppm)
  compounds_detail_res$mz_max[i]=ifelse(compounds_detail_res$exact_mass[i]>400,
                                        compounds_detail_res$exact_mass[i]+ppm*compounds_detail_res$exact_mass[i]/1000000,
                                        compounds_detail_res$exact_mass[i]+0.0004*ppm)
}

rownames(compounds_detail_res)=compounds_detail_res$id

# Save KEGG pathway and compound annotation objects.
save.image(file="kegg_pathway_info_detail.Rda")
save(compounds_detail_res,file="compounds_detail_res.Rda")
save(kegg_pathways,file="kegg_pathways.Rda")
save(kegg_pathways_id,file="kegg_pathways_id.Rda")

rm(list=ls())

load(file="compounds_detail_res.Rda")
load(file="kegg_pathways.Rda")
load(file="kegg_pathways_id.Rda")


# Retrieve KEGG reaction list.
reactions=keggList("Reaction")


ping=function(x,t){
  # Query a KEGG reaction with retry logic.
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
  # Parse one KEGG reaction into reactants, products, and enzymes.
  setTxtProgressBar(pb, x/length(reactions))

  request=ping(x,1)

  enzy=request$ENZYME
  temp=request$EQUATION

  # Alternative parsing from reaction description retained for compatibility.
  # temp=strsplit(reactions[x],"; ")[[1]]
  # temp=temp[length(temp)]

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


# Parse all KEGG reactions into component lists.
formulars=lapply(1:length(reactions),search_kegg_react)


# Build directed links between reactants, enzymes, and products.
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


# Save parsed reaction formulas and reaction-derived network links.
save(formulars,file="formulars.Rda")
save(links,file="links.Rda")
save(reactions,file="reactions.Rda")

# Export reaction-derived links as an Excel file.
openxlsx::write.xlsx(links,"links.xlsx")
