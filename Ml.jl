#Computes the log regression for a given chromosome
using Knet
include("log_regression.jl")

fileprefix = "chr1"
execute_log_regression(fileprefix)
