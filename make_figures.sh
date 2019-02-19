#!/usr/bin/env bash

gnuplot figures/plots/accuracy.gnuplot
gnuplot figures/plots/loss_basic_cnn.gnuplot
gnuplot figures/plots/loss_linear_regression.gnuplot

~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/accuracy.ps
~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/loss_basic_cnn.ps
~/workspace/binaries/epstopdf/epstopdf.pl figures/plots/loss_linear_regression.ps

