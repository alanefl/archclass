#!/usr/bin/env bash

gnuplot figures/plots/milestone/accuracy.gnuplot
gnuplot figures/plots/milestone/loss_basic_cnn.gnuplot
gnuplot figures/plots/milestone/loss_linear_regression.gnuplot

~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/milestone/accuracy.ps
~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/milestone/loss_basic_cnn.ps
~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/milestone/loss_linear_regression.ps

rm figures/plots/milestone/accuracy.ps
rm figures/plots/milestone/loss_basic_cnn.ps
rm figures/plots/milestone/loss_linear_regression.ps
