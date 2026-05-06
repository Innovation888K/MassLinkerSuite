

args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1] 
outputs=args[2]
pol=args[3]
#pol_init=args[4]
#file_name="QCP11.mzXML"
#print(pol)
root="D:\\"
for (f in list.files(work_dir)) {
  file_name=paste0(work_dir,"\\",f)
  load(file=paste0(root,"\\compounds_detail_res.Rda"))
  load(file=paste0(root,"\\kegg_pathways.Rda"))
  load(file=paste0(root,"\\kegg_pathways_id.Rda"))
  source(paste0(root,"\\RBF_fit.R"))
  output_folder=paste0(outputs)
  if(pol=="positive"){
    compounds_detail_res$mz_min=compounds_detail_res$mz_min+1.007825
    compounds_detail_res$mz_max=compounds_detail_res$mz_max+1.007825
  }else if(pol=="negative"){
    compounds_detail_res$mz_min=compounds_detail_res$mz_min-1.007825
    compounds_detail_res$mz_max=compounds_detail_res$mz_max-1.007825
  }
  
  if(!dir.exists(output_folder)){
    dir.create(output_folder, showWarnings = FALSE)
  }
  setwd(output_folder)
  file1=mzR::openMSfile(file_name)
  file2=
    MSnbase::readMSData(
      files = file_name,
      msLevel. = NULL,
      mode = "onDisk",
      verbose = TRUE
    )
  peaks = ProtGenerics::peaks(object = file1)
  rt = ProtGenerics::rtime(object = file2)
  rt=(rt-rt[1])/rt[length(rt)]*1800
  # if(pol_init=="High"){
  #   rt=rt
  # }else if(pol_init=="Low"){
  #   rt=1800-rt
  # }
  rm(list=c("file1", "file2"))
  dir.create(paste0(output_folder,"\\mz_zip"),showWarnings = FALSE)
  
  
  pb <- txtProgressBar(style = 3)
  pathway_zip=lapply(1:length(kegg_pathways),function(i){
    setTxtProgressBar(pb, i / length(kegg_pathways))
    pathway_compounds=kegg_pathways[[i]]
    pathway_compounds_detail=pathway_compounds[,1][pathway_compounds[,1]%in%compounds_detail_res$id]
    pathway_compounds_detail_res=compounds_detail_res[pathway_compounds_detail,]
    mz_zip=lapply(1:length(peaks),function(k){
      x=peaks[[k]]
      ret=list()
      x=data.frame(x)
      if(nrow(pathway_compounds_detail_res)>0){
        for (j in 1:nrow(pathway_compounds_detail_res)) {
          temp_detail_res=pathway_compounds_detail_res[j,]
          temp_peaks=x[x[,1]<temp_detail_res$mz_max&x[,1]>temp_detail_res$mz_min,]
          ret[[j]]=sum(temp_peaks$intensity[temp_peaks$intensity>1000])
        }
      }
      
      return(do.call(cbind,ret))
    })
    mz_zip_frame=data.frame(do.call(rbind,mz_zip))
    #save(mz_zip_frame,file=paste0(paste0(output_folder,"\\mz_zip\\"),kegg_pathways_id[i],"_mz_zip.Rda"))
    x=rt
    if(ncol(mz_zip_frame)>0){
      zipped_MSI_Image=do.call(cbind,lapply(1:ncol(mz_zip_frame),function(j){
        y=mz_zip_frame[,j]
        # if(sum(y)==0){
        #   rbf_model=list(centers=0,weights=0,sigma=0)
        # }else{
          rbf_model=rbf_fit(x,y,20,30,c(5,60),max_iterations = 10)
        # }
        rbf_model[["centers"]]=data.frame(rbf_model[["centers"]])
        rbf_model[["weights"]]=data.frame(rbf_model[["weights"]])
        rbf_model[["sigma"]]=data.frame(rbf_model[["sigma"]])
        ret=do.call(cbind,rbf_model)
        #colnames(ret)=paste0(kegg_pathways_id[i],"_",pathway_compounds_detail[i],"_",c("weights",'centers','sigmas'))
        #由于以上错误早期数据的标注是有误的，但仅限标注有误
        colnames(ret)=paste0(kegg_pathways_id[i],"_",pathway_compounds_detail[j],"_",c("weights",'centers','sigmas'))
        return(ret)
      }))
      return(zipped_MSI_Image)
    }
    return(NA)
  })
  
  f_name=strsplit(file_name,"\\\\")[[1]]
  f_name=strsplit(f_name[length(f_name)],"\\.")[[1]][1]
  dir.create(paste0(output_folder,"\\",f_name), showWarnings = FALSE)
  setwd(paste0(output_folder,"\\",f_name))
  
  
  
  
  wb <- openxlsx::createWorkbook()
  for (i in 1:271) {
    openxlsx::addWorksheet(wb, kegg_pathways_id[i])
    openxlsx::writeData(wb, sheet = i, pathway_zip[[i]])
  }
  openxlsx::saveWorkbook(wb, file = paste0(f_name, '.xlsx'), overwrite = TRUE)
}


















