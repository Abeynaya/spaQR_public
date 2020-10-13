
FILE[0 ]="../mats/advdiff/hc/advdiff_hc_d2_n127_q1_rho10_sig2.mm"
FILE[1 ]="../mats/advdiff/hc/advdiff_hc_d2_n255_q1_rho10_sig2.mm"
# FILE[2 ]="../mats/advdiff/hc/advdiff_hc_d2_n511_q1_rho10_sig2.mm"
# FILE[3 ]="../mats/advdiff/hc/advdiff_hc_d2_n1023_q1_rho10_sig2.mm"

N=(127 255 511 1023)
T=(1e-3 1e-5)

for i in {0..3}
do
n=$((($i)/2))
t=$((($i)%2))
# srun ../spaQR matrix mesh_size dimension tolerance skip scale solver 
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 --scale 1 --solver "GMRES"
.././spaQR -m ${FILE[n]} -n ${N[n]} -d 2  -t ${T[t]}  --skip 4 --scale 0 --solver "GMRES"
done

