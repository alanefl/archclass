set term postscript eps enhanced color font "Helvetica,15" size 6,4
set output "figures/plots/milestone/accuracy.ps"

# We are reading from CSV
set datafile separator comma

# Margin specifications
set tmargin 5
set bmargin 5
set lmargin 10
set rmargin 10

# Title, Key, Labels, general formatting
set grid
set ylabel "Accuracy"
set xlabel "Training Epoch"
set yrange [-.3:1]
set key bottom


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


plot "figures/data/milestone/multinomial_logistic_regression.csv" using 1:2 with linespoints ls 1 title "Multinomial Logistic Regression | Train", \
  "figures/data/milestone/multinomial_logistic_regression.csv" using 1:4 with linespoints ls 2 title "Multinomial Logistic Regression | Dev", \
  "figures/data/milestone/basic_cnn_no_regularization.csv" using 1:2 with linespoints ls 3 title "Baseline CNN No Regularization | Train", \
  "figures/data/milestone/basic_cnn_no_regularization.csv" using 1:4 with linespoints ls 4 title "Baseline CNN No Regularization | Dev", \
  "figures/data/milestone/basic_cnn_moderate_dropout.csv" using 1:2 with linespoints ls 5 title "Baseline CNN w/ moderate Dropout | Train", \
  "figures/data/milestone/basic_cnn_moderate_dropout.csv" using 1:4 with linespoints ls 6 title "Baseline CNN w/ moderate Dropout  | Dev", \
  "figures/data/milestone/basic_cnn_heavy_dropout.csv" using 1:2 with linespoints ls 7 title "Baseline CNN w/ heavy Dropout | Train", \
  "figures/data/milestone/basic_cnn_heavy_dropout.csv" using 1:4 with linespoints ls 8 title "Baseline CNN w/ heavy Dropout  | Dev"