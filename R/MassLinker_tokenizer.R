# MassLinker tokenization script for LC-MS raw files.
#
# This script converts raw LC-MS files into MassLinker tokenized Excel outputs.
# For each input file, the script maps MS peaks to KEGG-associated compound
# m/z windows, constructs intensity traces along the retention-time axis, and
# fits each trace with a fixed number of radial basis functions (RBFs).
#
# The output for each sample is an Excel workbook. Each worksheet corresponds
# to one KEGG pathway, and each compound-associated feature is represented by
# RBF-derived parameters: weights, centers, and sigmas.
#
# Command-line arguments:
#   args[1] : work_dir
#       Directory containing raw LC-MS files.
#   args[2] : outputs
#       Output directory for MassLinker tokenized Excel files.
#   args[3] : pol
#       Ionization polarity. Supported values are "positive" and "negative".
#   args[4] : root
#       Root directory containing compound/pathway annotations and RBF_fit.R.

args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1] 
outputs=args[2]
pol=args[3]

# Optional initial polarity argument retained for compatibility.
# pol_init=args[4]

# Example filename retained for debugging.
# file_name="QCP11.mzXML"

# Print polarity for debugging if needed.
# print(pol)

root=args[4]

# Process each raw LC-MS file in the working directory.
for (f in list.files(work_dir)) {
  file_name=paste0(work_dir,"\\",f)

  # Load compound annotation and KEGG pathway metadata.
  load(file=paste0(root,"\\compounds_detail_res.Rda"))
  load(file=paste0(root,"\\kegg_pathways.Rda"))
  load(file=paste0(root,"\\kegg_pathways_id.Rda"))

  # Load the RBF fitting function.
  source(paste0(root,"\\RBF_fit.R"))

  output_folder=paste0(outputs)

  # Adjust m/z windows according to ionization polarity.
  # Positive mode adds proton mass; negative mode subtracts proton mass.
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

  # Open the raw MS file and read MS data in on-disk mode.
  file1=mzR::openMSfile(file_name)
  file2=
    MSnbase::readMSData(
      files = file_name,
      msLevel. = NULL,
      mode = "onDisk",
      verbose = TRUE
    )

  # Extract peak lists and retention times.
  peaks = ProtGenerics::peaks(object = file1)
  rt = ProtGenerics::rtime(object = file2)

  # Normalize retention time to a 0-1800 scale.
  rt=(rt-rt[1])/rt[length(rt)]*1800

  # Optional retention-time orientation correction retained for compatibility.
  # if(pol_init=="High"){
  #   rt=rt
  # }else if(pol_init=="Low"){
  #   rt=1800-rt
  # }

  rm(list=c("file1", "file2"))

  dir.create(paste0(output_folder,"\\mz_zip"),showWarnings = FALSE)
  
  # Iterate through all KEGG pathways and build RBF-derived token features.
  pb <- txtProgressBar(style = 3)
  pathway_zip=lapply(1:length(kegg_pathways),function(i){
    setTxtProgressBar(pb, i / length(kegg_pathways))

    pathway_compounds=kegg_pathways[[i]]

    # Select pathway compounds that have available compound-detail annotations.
    pathway_compounds_detail=pathway_compounds[,1][pathway_compounds[,1]%in%compounds_detail_res$id]
    pathway_compounds_detail_res=compounds_detail_res[pathway_compounds_detail,]

    # Construct intensity traces for all compounds in the current pathway.
    mz_zip=lapply(1:length(peaks),function(k){
      x=peaks[[k]]
      ret=list()
      x=data.frame(x)

      if(nrow(pathway_compounds_detail_res)>0){
        for (j in 1:nrow(pathway_compounds_detail_res)) {
          temp_detail_res=pathway_compounds_detail_res[j,]

          # Select peaks within the compound-specific m/z window and sum
          # high-intensity signals.
          temp_peaks=x[x[,1]<temp_detail_res$mz_max&x[,1]>temp_detail_res$mz_min,]
          ret[[j]]=sum(temp_peaks$intensity[temp_peaks$intensity>1000])
        }
      }
      
      return(do.call(cbind,ret))
    })

    mz_zip_frame=data.frame(do.call(rbind,mz_zip))

    # Optional intermediate saving retained for compatibility.
    # save(mz_zip_frame,file=paste0(paste0(output_folder,"\\mz_zip\\"),kegg_pathways_id[i],"_mz_zip.Rda"))

    x=rt

    if(ncol(mz_zip_frame)>0){
      zipped_MSI_Image=do.call(cbind,lapply(1:ncol(mz_zip_frame),function(j){
        y=mz_zip_frame[,j]

        # Fit the chromatographic intensity trace using 20 RBF components.
        # if(sum(y)==0){
        #   rbf_model=list(centers=0,weights=0,sigma=0)
        # }else{
          rbf_model=rbf_fit(x,y,20,30,c(5,60),max_iterations = 10)
        # }

        rbf_model[["centers"]]=data.frame(rbf_model[["centers"]])
        rbf_model[["weights"]]=data.frame(rbf_model[["weights"]])
        rbf_model[["sigma"]]=data.frame(rbf_model[["sigma"]])

        # Combine RBF parameters into one feature block.
        ret=do.call(cbind,rbf_model)

        # Original column-name construction retained for compatibility.
        # colnames(ret)=paste0(kegg_pathways_id[i],"_",pathway_compounds_detail[i],"_",c("weights",'centers','sigmas'))

        # Earlier outputs used incorrect annotation indexing; only labels were affected.
        colnames(ret)=paste0(kegg_pathways_id[i],"_",pathway_compounds_detail[j],"_",c("weights",'centers','sigmas'))

        return(ret)
      }))

      return(zipped_MSI_Image)
    }

    return(NA)
  })
  
  # Derive the sample name from the raw file name.
  f_name=strsplit(file_name,"\\\\")[[1]]
  f_name=strsplit(f_name[length(f_name)],"\\.")[[1]][1]

  # Create one output folder per sample.
  dir.create(paste0(output_folder,"\\",f_name), showWarnings = FALSE)
  setwd(paste0(output_folder,"\\",f_name))
  
  # Save tokenized pathway-level features into an Excel workbook.
  wb <- openxlsx::createWorkbook()

  for (i in 1:271) {
    openxlsx::addWorksheet(wb, kegg_pathways_id[i])
    openxlsx::writeData(wb, sheet = i, pathway_zip[[i]])
  }

  openxlsx::saveWorkbook(wb, file = paste0(f_name, '.xlsx'), overwrite = TRUE)
}
