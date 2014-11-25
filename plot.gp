set term jpeg
set output "Results.jpg"
set title "Comparing Serial vs Parallel"
set xlabel "Number Of Elelments"
set ylabel "Training Time (s)"
set yrange [18:25]
plot "serial.dat" t "Serial" lt 1 lc 1 lw 3,\
"parallel.dat" t "Parallel" lt 2 lc 2 lw 2