# install_R_dependencies.R
# Install R dependencies required by MassLinker Suite.
#
# Required packages:
# - pracma: numerical routines used for signal fitting and matrix operations
# - mzR: mzML/mzXML mass spectrometry data access
# - KEGGREST: KEGG pathway and compound metadata retrieval
# - tcltk: GUI/file-dialog support, usually included with standard R installations

message("Installing R dependencies for MassLinker Suite...")

# Install CRAN dependency
if (!requireNamespace("pracma", quietly = TRUE)) {
  install.packages("pracma")
} else {
  message("pracma is already installed.")
}

# Install Bioconductor manager if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Install Bioconductor dependencies
bioc_packages <- c("mzR", "KEGGREST")

for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    message(pkg, " is already installed.")
  }
}

# Check tcltk availability
message("Checking tcltk availability...")

if ("tcltk" %in% rownames(installed.packages())) {
  suppressWarnings({
    tcltk_available <- capabilities("tcltk")
  })

  if (isTRUE(tcltk_available)) {
    message("tcltk is available.")
  } else {
    warning(
      "tcltk is installed but Tcl/Tk capability is not available. ",
      "Please install system Tcl/Tk libraries or use a Tcl/Tk-enabled R distribution."
    )
  }
} else {
  warning(
    "tcltk was not found. It is usually included with standard R installations. ",
    "Please check your R installation."
  )
}

message("R dependency installation finished.")
