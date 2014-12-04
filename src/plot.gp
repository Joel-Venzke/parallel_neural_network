f(x) = a*x**2+b
f2(x) = a1*x**2+b1
fit f(x) 'data/serial.dat' using 1:2 via a, b
fit f2(x) 'data/parallel.dat' using 1:2 via a1, b1
set term jpeg size 900,500
set output "Results.jpg"
set title "Comparing Training Time on Neural Networks"
set xlabel "Number Of Elelments"
set ylabel "Training Time (s)"
#set xrange [0:1200]
set key top left
#set yrange [-5:300]
plot "data/serial.dat" t "Serial" lt 1 lc 1 lw 3, f(x) t 'x^2' lc 1,\
"data/parallel.dat" t "Parallel" lt 2 lc 2 lw 2, f2(x) t 'x^2' lc 2