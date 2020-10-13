
FILE[0 ]="../mats/suite_sparse/cavity15.mtx"
FILE[1 ]="../mats/suite_sparse/cavity26.mtx"
FILE[2 ]="../mats/suite_sparse/dw4096.mtx"
FILE[3 ]="../mats/suite_sparse/Goodwin_030.mtx"
FILE[4 ]="../mats/suite_sparse/inlet.mtx"
FILE[5 ]="../mats/suite_sparse/Goodwin_040.mtx"
FILE[6 ]="../mats/suite_sparse/wang4.mtx"
FILE[7 ]="../mats/suite_sparse/Zhao1.mtx"
FILE[8 ]="../mats/suite_sparse/Chevron1.mtx"
FILE[9 ]="../mats/suite_sparse/cz40948.mtx"

T=(1e-3 1e-6)

for i in {0..19}
do
n=$((($i)/2))
t=$((($i)%2))
.././spaQR -m ${FILE[n]} -t ${T[t]} --hsl 0 --skip 3 --scale 1 --solver "GMRES"
.././spaQR -m ${FILE[n]} -t ${T[t]} --hsl 0 --skip 3 --scale 0 --solver "GMRES"
done

