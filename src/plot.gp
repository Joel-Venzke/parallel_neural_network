set term jpeg size 900,500
set output "Results.jpg"
set title "Comparing Serial vs Parallel"
set xlabel "Number Of Elelments"
set ylabel "Training Time (s)"
plot "data/serial.dat" t "Serial" lt 1 lc 1 lw 3,\
"data/parallel.dat" t "Parallel" lt 2 lc 2 lw 2