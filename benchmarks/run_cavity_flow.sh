
FILE[0 ]="../mats/cavity/e40r0000.mtx"
FILE[1 ]="../mats/cavity/e40r1000.mtx"
FILE[2 ]="../mats/cavity/e40r5000.mtx"
# FILE[3 ]="../mats/cavity/e40r3000.mtx"
# FILE[4 ]="../mats/cavity/e40r0100.mtx"
# FILE[5 ]="../mats/cavity/e40r0500.mtx"

T=(1e-5)

for i in {0..2}
do
n=$((($i)/1))
t=$((($i)%1))
.././spaQR -m ${FILE[n]} -t ${T[t]} --skip 3 --scale 1 --solver "GMRES"
.././spaQR -m ${FILE[n]} -t ${T[t]} --skip 3 --scale 0 --solver "GMRES"
done

