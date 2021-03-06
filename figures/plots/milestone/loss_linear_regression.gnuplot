set term postscript eps enhanced color font "Helvetica,15" size 6,4
set output "figures/plots/milestone/loss_linear_regression.ps"

# We are reading from CSV
set datafile separator comma

# Margin specifications
set tmargin 5
set bmargin 5
set lmargin 10
set rmargin 10

# Title, Key, Labels, general formatting
set grid
set ylabel "Loss"
set xlabel "Training Epoch"
set key top right

# Simple line chart
set style line 1 linecolor rgb '#e6194b' lt 1 linewidth 3 pointtype 6
set style line 2 linecolor rgb '#e6194b' lt 1 linewidth 3 pointtype 6 dashtype 3
set style line 3 linecolor rgb '#3cb44b' lt 1 linewidth 3 pointtype 6
set style line 4 linecolor rgb '#3cb44b' lt 1 linewidth 3 pointtype 6 dashtype 3
set style line 5 linecolor rgb '#ffe119' lt 1 linewidth 3 pointtype 6
set style line 6 linecolor rgb '#ffe119' lt 1 linewidth 3 pointtype 6 dashtype 3
set style line 7 linecolor rgb '#4363d8' lt 1 linewidth 3 pointtype 6
set style line 8 linecolor rgb '#4363d8' lt 1 linewidth 3 pointtype 6 dashtype 3
set style line 9 linecolor rgb '#f58231' lt 1 linewidth 3


plot "figures/data/milestone/multinomial_logistic_regression.csv" using 1:3 with linespoints ls 1 title "Multinomial Logistic Regression | Train", \
  "figures/data/milestone/multinomial_logistic_regression.csv" using 1:5 with linespoints ls 2 title "Multinomial Logistic Regression | Dev", \