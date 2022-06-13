import numpy as np
import matplotlib.pyplot as plt
ordre= 2
inc_face= (ordre+1)*2
maillage =np.array([2,3,5,10,15,20,25,30,35,40,45,50,55,60] )
maillage =np.array([2,3,5,10,15,20,25,30,35,40,45,50,55] )
inc=[]
for i in range(0,len(maillage)):
    faces= (maillage[i]-1)*4+2*(maillage[i]-2)*(maillage[i]-1)
    inc.append(faces*inc_face)

    print ('Sur un maillage de taille',maillage[i],'On a ',faces, 'faces, et donc ',inc[i], 'inconnues')
# deplacement = np.array([0.00491496,0.00651321,0.00675453, 0.00691146 ,0.00694351,0.00695848,0.00696756,0.00697367,0.00697833,0.00698184,
#                   0.00698435,0.00698667,0.00698960,0.00698992])
deplacement = np.array([0.00491496,0.00651321,0.00675453, 0.00691146 ,0.00694351,0.00695848,0.00696756,0.00697367,0.00697833,0.00698184,
                  0.00698435,0.00698667,0.00698960])

deplacement1 = np.array([0.00428209,0.00577132,0.0065870, 0.0068524 ,0.00691182,0.00694536,0.00695595,0.00696528,
                        0.00697175,0.00697175,0.00698229,0.00698413,0.00698687,])

deplacement3 = np.array([0.00677245, 0.00665421 ,0.00686445,0.00693885,0.00696008,0.00697343,0.00698054,0.00622757,
                        0.00698835,0.00699125,0.00699336,0.00699513,0.00623786,])

# Q1
Q1DATA = np.array([[7.68098057277, 0.164948453608],
[17.7827941004, 0.412371134021],
[49.8788769645, 1.01030927835],
[169.498815139, 2.35051546392],
[604.296390238, 4.0824742268],
[2371.37370566, 5.64948453608],
[9531.61883235, 6.49484536082],
[36517.4127255, 6.80412371134]])

Q2DATA = np.array([
[16.7291966358,  0.729166666667],
[41.7339691542,  3.70833333333],
[123.879168545,  6.04166666667],
[432.788560749,  6.64583333333],
[1554.4643128 , 6.91666666667],
[6156.07813934,  6.95833333333],
[25565.0147781,  7],
[98868.9990676,  7.04166666667],
])

Q1X = Q1DATA[:, 0]
Q1Y = Q1DATA[:, 1]
Q2X = Q2DATA[:, 0]
Q2Y = Q2DATA[:, 1]

hho22_x = np.array(
    [
24.0681826687,
73.4478997218,
246.463480571,
909.416619431,
3436.23197098,
13615.0931992,
53945.9397355,
# 213745.464049,
    ]
)
hho22_y = np.array(
    [
3.78835978836,
6.4126984127,
6.6455026455,
6.89947089947,
6.98412698413,
7.00529100529,
7.02645502646,
# 7.06878306878,
    ]
)

#print(inc[13])
plt.grid(True)
plt.xlim(1.e1,1.e5)
plt.ylim(0,8)
plt.xscale('log')
# plt.loglog(inc,deplacement)
plt.plot(inc, 1.e3 * deplacement, color="blue")
plt.scatter(inc,1.e3 * deplacement, color="blue", label="HHO(2,2) python implementation")
plt.plot(inc, 1.e3 * deplacement1, color="cyan")
plt.scatter(inc,1.e3 * deplacement1, color="cyan", label="HHO(1,1) python implementation")
# plt.plot(np.array(inc), 1.e3 * deplacement3, color="magenta")
# plt.scatter(np.array(inc), 1.e3 * deplacement3, color="magenta", label="HHO(3,3) python implementation")
plt.plot(hho22_x, hho22_y, color="red")
plt.scatter(hho22_x, hho22_y, color="red", label="HHO(2,2) [2019, Abbas, Ern, Pignet]")
plt.plot(Q1X, Q1Y, color="green")
plt.scatter(Q1X, Q1Y, color="green", label="QUA4 [2019, Abbas, Ern, Pignet]")
plt.plot(Q2X, Q2Y, color="purple")
plt.scatter(Q2X, Q2Y, color="purple", label="QUA8 [2019, Abbas, Ern, Pignet]")


plt.xlabel ("number of DOFs")
plt.ylabel("Vertical displacement of point A [mm]")
plt.legend()
#
# plt.title("Etude du deplacement du point A en fonction du nombre d'inconnues Hdg_equal ordre 2" )

plt.show()