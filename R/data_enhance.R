library(mzR)
args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1]
output <- args[2]
enhance_num <- as.integer(args[3])
dir.create(output, showWarnings = FALSE, recursive = TRUE)
set.seed(42)
files <- list.files(
  work_dir,
  pattern = "\\.(mzXML|mzML|cdf|CDF)$",
  full.names = TRUE
)
for (file_name in files) {
  message("Processing: ", file_name)
  file1 <- mzR::openMSfile(file_name)
  header <- mzR::header(file1)
  peaks <- ProtGenerics::peaks(object = file1)
  rt <- header$retentionTime
  rt_diff <- rnorm(enhance_num, mean = 0, sd = 10)
  for (i in seq_len(enhance_num)) {
    augmented_peaks <- lapply(peaks, function(x) {
      x <- data.frame(x)
      if (nrow(x) == 0) {
        return(as.matrix(x))
      }
      mz_shift <- rnorm(
        nrow(x),
        mean = 0,
        sd = 5
      ) * ifelse(x$mz < 400, 400, x$mz) / 1000000
      x$mz <- x$mz + mz_shift
      int_shift <- rnorm(nrow(x), mean = 0, sd = 0.06)
      int_shift <- pmax(pmin(int_shift, 0.3), -0.3)
      x$intensity <- x$intensity * (1 - int_shift)
      x$intensity[x$intensity < 0] <- 0
      return(as.matrix(x))
    })
    header_i <- header
    header_i$retentionTime <- rt + rt_diff[i]
    header_i$retentionTime[header_i$retentionTime < 0] <- 0
    sample_name <- basename(file_name)
    output_file <- file.path(output, paste0(i, "-", sample_name))
    message("Writing: ", output_file)
    mzR::writeMSData(
      object = augmented_peaks,
      file = output_file,
      header = header_i
    )
  }
  mzR::close(file1)
}
