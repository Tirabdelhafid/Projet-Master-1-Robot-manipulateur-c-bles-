# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:38:48 2022

@author: hafid
"""
#importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

#entrés = Position du centre de la plateforme selon X, Y, Z initials
         # et matrice d'orientation du plateforme mobile 
p = np.array( [ 10.0 , 10.0, 10.0 ] )
théta = 0.0
psi = 0.0
phi = 0.0
R = np.array( [ [math.cos(théta)*math.cos(psi),-math.cos(théta)*math.sin(psi), math.sin(théta)] 
               ,[math.cos(phi)*math.sin(psi) + math.sin(phi)*math.sin(théta)*math.cos(psi),math.cos(phi)*math.cos(psi)-math.sin(phi)*math.sin(théta)*math.sin(psi) , -math.sin(phi)*math.cos(théta)] 
               ,[math.sin(phi)*math.sin(psi)-math.cos(phi)*math.sin(théta)*math.cos(psi),math.sin(phi)*math.cos(psi)+math.cos(phi)*math.sin(théta)*math.sin(psi),math.cos(phi)*math.cos(théta)]
             ] )
#désignation des variables :
#position de la sortie du cables
a1 = np.array([0.0,10.0,10.0])
a2 = np.array([10.0,10.0,10.0])
a3 = np.array([10.0,0.0,10.0])
a4 = np.array([0.0,0.0,10.0])
a5 = np.array([0.0,10.0,10.0])
a6 = np.array([10.0,10.0,10.0])
a7 = np.array([10.0,0.0,10.0])
a8 = np.array([0.0,0.0,10.0])
#point d'attache du cables dans le repère du platforme
b1 = np.array([-1.0,1.0,1.0])
b2 = np.array([1.0,1.0,1.0])
b3 = np.array([1.0,-1.0,1.0])
b4 = np.array([-1.0,-1.0,1.0])
b5 = np.array([-1.0,1.0,-1.0])
b6 = np.array([1.0,1.0,-1.0])
b7 = np.array([1.0,-1.0,-1.0])
b8 = np.array([-1.0,-1.0,-1.0])
#Sorties = vecteur linéaire le long des cables
s1 = p + np.dot(R,b1) - a1
s2 = p + np.dot(R,b2) - a2
s3 = p + np.dot(R,b3) - a3
s4 = p + np.dot(R,b4) - a4
s5 = p + np.dot(R,b5) - a5
s6 = p + np.dot(R,b6) - a6
s7 = p + np.dot(R,b7) - a7
s8 = p + np.dot(R,b8) - a8
#Calcule des longeurs des cables
Length_s1 = np.sqrt(s1[0]**2 + s1[1]**2 + s1[2]**2)
Length_s2 = np.sqrt(s2[0]**2 + s2[1]**2 + s1[2]**2)
Length_s3 = np.sqrt(s3[0]**2 + s3[1]**2 + s3[2]**2)
Length_s4 = np.sqrt(s4[0]**2 + s4[1]**2 + s4[2]**2)
Length_s5 = np.sqrt(s5[0]**2 + s5[1]**2 + s5[2]**2)
Length_s6 = np.sqrt(s6[0]**2 + s6[1]**2 + s6[2]**2)
Length_s7 = np.sqrt(s7[0]**2 + s7[1]**2 + s7[2]**2)
Length_s8 = np.sqrt(s8[0]**2 + s8[1]**2 + s8[2]**2)
# print("Length_s1 = ",Length_s1,"Length_s2 = ",Length_s2,
#       "Length_s3 = ",Length_s3,"Length_s4 = ",Length_s4,
#       "Length_s5 = ",Length_s5,"Length_s6 = ",Length_s6,
#       "Length_s7 = ",Length_s7,"Length_s8 = ",Length_s8)


# plot du robot parallèle à cables
x = [ a1[0],s1[0]+a1[0],s2[0]+a2[0],a2[0],s2[0]+a2[0],a3[0]+s3[0],a3[0],
      s3[0]+a3[0],a4[0]+s4[0],a4[0],s4[0]+a4[0], a1[0]+s1[0],
      s5[0]+a5[0],a5[0],s5[0]+a5[0],s6[0]+a6[0],a6[0],s6[0]+a6[0],
      s7[0]+a7[0],a7[0],s7[0]+a7[0],s8[0]+a8[0],a8[0],s8[0]+a8[0],s4[0]+
      a4[0],s8[0]+a8[0],s5[0]+a5[0],s8[0]+a8[0],
      s7[0]+a7[0],s3[0]+a3[0],s7[0]+a7[0],s6[0]+a6[0],s2[0]+a2[0]]
y =  [  a1[1],s1[1]+a1[1],s2[1]+a2[1],a2[1],s2[1]+a2[1],a3[1]+s3[1],a3[1],
      s3[1]+a3[1],a4[1]+s4[1],a4[1],s4[1]+a4[1], a1[1]+s1[1],
      s5[1]+a5[1],a5[1],s5[1]+a5[1],s6[1]+a6[1],a6[1],s6[1]+a6[1],
      s7[1]+a7[1],a7[1],s7[1]+a7[1],s8[1]+a8[1],a8[1],s8[1]+a8[1],s4[1]
      +a4[1],s8[1]+a8[1],s5[1]+a5[1],s8[1]+a8[1],
      s7[1]+a7[1],s3[1]+a3[1],s7[1]+a7[1],s6[1]+a6[1],s2[1]+a2[1] ]
z = [  a1[2],s1[2]+a1[2],s2[2]+a2[2],a2[2],s2[2]+a2[2],a3[2]+s3[2],a3[2],
      s3[2]+a3[2],a4[2]+s4[2],a4[2],s4[2]+a4[2], a1[2]+s1[2],
      s5[2]+a5[2],a5[2],s5[2]+a5[2],s6[2]+a6[2],a6[2],s6[2]+a6[2],
      s7[2]+a7[2],a7[2],s7[2]+a7[2],s8[2]+a8[2],a8[2],s8[2]+a8[2],s4[2]
      +a4[2],s8[2]+a8[2],s5[2]+a5[2],s8[2]+a8[2],
      s7[2]+a7[2],s3[2]+a3[2],s7[2]+a7[2],s6[2]+a6[2],s2[2]+a2[2]  ]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(x, y, z, label='Robot 6ddl - 8 cables')
# # Set axes label
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# plt.show()

#fonction pour le calcul de produit vectoriel
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c
# création matrice jacobienne inverse
u1 = s1/Length_s1
u2 = s2/Length_s2
u3 = s3/Length_s3
u4 = s4/Length_s4
u5 = s5/Length_s5
u6 = s6/Length_s6
u7 = s7/Length_s7
u8 = s8/Length_s8
b1_u1 = cross(b1,u1)
b2_u2 = cross(b2,u2)
b3_u3 = cross(b3,u3)
b4_u4 = cross(b4,u4)
b5_u5 = cross(b5,u5)
b6_u6 = cross(b6,u6)
b7_u7 = cross(b7,u7)
b8_u8 = cross(b8,u8)
J_1 = np.bmat([  [[u1, u2, u3, u4, u5, u6, u7, u8],
                  [b1_u1,b2_u2,b3_u3,b4_u4,b5_u5,b6_u6,b7_u7,b8_u8] ]     ])

# controle vitesses articulaires de la plateforme
s_1 , s_2 , s_3, s_4, s_5, s_6, s_7, s_8 =  [],[],[],[],[],[],[],[]
n_1, n_2, n_3,n_4,n_5,n_6,n_7, n_8 = [], [],[], [], [], [], [], []
# définition des listes pour le traçage de la trajectoire de la plateforme mobile
px, py, pz = [],[],[]
#♣ défintion des listes pour le traçage des courbes d'évolution des longeurs des cables
Ls1, Ls2,Ls3,Ls4,Ls5,Ls6,Ls7,Ls8 = [],[],[],[],[],[],[],[]

#########################################################
#      Choix de la position et l'orientation désiré
#########################################################
x2 = 5
y2 = 15
z2 = 10
phi_fin = 0.0   #selon axe X
théta_fin = 0.0 #selon axe y
psi_fin = 0.0   #selon axe Z

##########################################################
#      choix de vitesse de translation et rotation désirés
##########################################################
v = 80
phi_point = 0.0
théta_point = 0.0
psi_point = 0.0

########################################################
#                      Animation
########################################################
i = 0 #nombre d'itération
# augmenter nombre d'itération si nécessaire  i<20
while( i<20 and (p[0]<=x2 or p[0]>=x2) and (p[1]<=y2 or p[1]>=y2) and (p[2]<=z2 or p[2]>=z2)
      and (phi<=phi_fin or phi>=phi_fin) and  (psi<=psi_fin or psi>=psi_fin) and (théta<=théta_fin or théta>=théta_fin)):    
    i = i + 1
    #position de la sortie du cables
    a1 = np.array([0.0,20.0,20.0])
    a2 = np.array([20.0,20.0,20.0])
    a3 = np.array([20.0,0.0,20.0])
    a4 = np.array([0.0,0.0,20.0])
    a5 = np.array([0.0,20.0,20.0])
    a6 = np.array([20.0,20.0,20.0])
    a7 = np.array([20.0,0.0,20.0])
    a8 = np.array([0.0,0.0,20.0])
    #point d'attache dans le repère du platforme
    b1 = np.array([-1.0,1.0,1.0])
    b2 = np.array([1.0,1.0,1.0])
    b3 = np.array([1.0,-1.0,1.0])
    b4 = np.array([-1.0,-1.0,1.0])
    b5 = np.array([-1.0,1.0,-1.0])
    b6 = np.array([1.0,1.0,-1.0])
    b7 = np.array([1.0,-1.0,-1.0])
    b8 = np.array([-1.0,-1.0,-1.0])
    #Sorties = vecteur linéaire le long des cables
    s1 = p + np.dot(R,b1) - a1
    s2 = p + np.dot(R,b2) - a2
    s3 = p + np.dot(R,b3) - a3
    s4 = p + np.dot(R,b4) - a4
    s5 = p + np.dot(R,b5) - a5
    s6 = p + np.dot(R,b6) - a6
    s7 = p + np.dot(R,b7) - a7
    s8 = p + np.dot(R,b8) - a8
    #calcule longeurs des cables
    Length_s1 = np.sqrt(s1[0]**2 + s1[1]**2 + s1[2]**2)
    Length_s2 = np.sqrt(s2[0]**2 + s2[1]**2 + s1[2]**2)
    Length_s3 = np.sqrt(s3[0]**2 + s3[1]**2 + s3[2]**2)
    Length_s4 = np.sqrt(s4[0]**2 + s4[1]**2 + s4[2]**2)
    Length_s5 = np.sqrt(s5[0]**2 + s5[1]**2 + s5[2]**2)
    Length_s6 = np.sqrt(s6[0]**2 + s6[1]**2 + s6[2]**2)
    Length_s7 = np.sqrt(s7[0]**2 + s7[1]**2 + s7[2]**2)
    Length_s8 = np.sqrt(s8[0]**2 + s8[1]**2 + s8[2]**2)
    # création matrice jacobienne inverse
    u1 = s1/Length_s1
    u2 = s2/Length_s2
    u3 = s3/Length_s3
    u4 = s4/Length_s4
    u5 = s5/Length_s5
    u6 = s6/Length_s6
    u7 = s7/Length_s7
    u8 = s8/Length_s8
    b1_u1 = cross(b1,u1)
    b2_u2 = cross(b2,u2)
    b3_u3 = cross(b3,u3)
    b4_u4 = cross(b4,u4)
    b5_u5 = cross(b5,u5)
    b6_u6 = cross(b6,u6)
    b7_u7 = cross(b7,u7)
    b8_u8 = cross(b8,u8)
    J_1 = np.bmat([  [[u1, u2, u3, u4, u5, u6, u7, u8],
                      [b1_u1,b2_u2,b3_u3,b4_u4,b5_u5,b6_u6,b7_u7,b8_u8] ]     ])
    
    #équation paramétrique selon X, Y, Z:
    # x(t) = (t*v*(x2-p[0]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    # y(t) = (t*v*(y2-p[1]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    # z(t) = (t*v*(z2-p[2]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    #vitesee de translation selon X, Y, Z:    
    x_point = (v*(x2-p[0]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    y_point = (v*(y2-p[1]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    z_point = (v*(z2-p[2]))/ np.sqrt( (x2-p[0])**2+(y2-p[1])**2+(z2-p[2])**2 )
    
    #création de vecteur paramètres opérationnelles:
    x_dot = np.array([[x_point,y_point,z_point,phi_point,théta_point,psi_point]])
    #trouver le vecteur des paramètre articulaire à partir du produit entre vecteur
    #paramètre opérationnels et la matrice Jacobienne inverse
    q = np.dot(J_1,x_dot.T)
    # print("vitesse d'enroulement cable 1 = ",q[0],
    #   "vitesse d'enroulement cable 2 = ",q[1],
    #   "vitesse d'enroulement cable 3 = ",q[2],
    #   "vitesse d'enroulement cable 4 = ",q[3],
    #   "vitesse d'enroulement cable 5 = ",q[4],
    #   "vitesse d'enroulement cable 6 = ",q[5],
    #   "vitesse d'enroulement cable 7 = ",q[6],
    #   "vitesse d'enroulement cable 8 = ",q[7])
    
    #définition de matrice de rotation:
    R = np.array( [ [math.cos(théta)*math.cos(psi),-math.cos(théta)*math.sin(psi), math.sin(théta)] 
                ,[math.cos(phi)*math.sin(psi) + math.sin(phi)*math.sin(théta)*math.cos(psi),math.cos(phi)*math.cos(psi)-math.sin(phi)*math.sin(théta)*math.sin(psi) , -math.sin(phi)*math.cos(théta)] 
                ,[math.sin(phi)*math.sin(psi)-math.cos(phi)*math.sin(théta)*math.cos(psi),math.sin(phi)*math.cos(psi)+math.cos(phi)*math.sin(théta)*math.sin(psi),math.cos(phi)*math.cos(théta)]
                ] )
    
    
    
    #calcul des tension dans les cables:
    # on définit la masse de la plateforme mobile et force de gravité:
    m =10 
    g =9.81
    #création de la matrice Jacobienne :
    J = np.bmat([  [[np.transpose(s1)/Length_s1, np.transpose(s2)/Length_s2, 
                     np.transpose(s3)/Length_s3, np.transpose(s4)/Length_s4, 
                     np.transpose(s5)/Length_s5, np.transpose(s6)/Length_s6, 
                     np.transpose(s7)/Length_s7, np.transpose(s8)/Length_s8],
                      [np.transpose(cross(np.dot(R,b1),(p-a1)))/Length_s1,
                       np.transpose(cross(np.dot(R,b2),(p-a2)))/Length_s2,
                       np.transpose(cross(np.dot(R,b3),(p-a3)))/Length_s3,
                       np.transpose(cross(np.dot(R,b4),(p-a4)))/Length_s4,
                       np.transpose(cross(np.dot(R,b5),(p-a5)))/Length_s5,
                       np.transpose(cross(np.dot(R,b6),(p-a6)))/Length_s6,
                       np.transpose(cross(np.dot(R,b7),(p-a7)))/Length_s7, 
                       np.transpose(cross(np.dot(R,b8),(p-a8)))/Length_s8] ]    
                 ])
    #création de la matrice des torseurs
    w = - J.T
    wt = np.transpose(w)
    #création du vecteur d'efforts appliqué sur la plateforme
    f = np.array([0,0,m*g,0,0,0])
    #vecteur des efforts = matrice des torseurs * vecteur des tensions des cables
    #on multipie l'équation par le transposé de la matrice des torseurs et on résoudre l'équation matricielle avec
    #le vecteur de tension comme inconnu
    #et à l'aide d'alogorithme scipy.optimize on trouve la bonne distribution des tensions des 8 cables, sous contrainte
    #de trouver ses valeurs en signe positif
    wt_w = np.dot(wt,w)
    wt_f = np.dot(wt,f)


    # trouver une solution particulière à l'aide de module np.linalg.lstsq :
    
    # wt = np.transpose(w)
    # f = np.array([0,0,m*g,0,0,0])
    # def calcul_tension(m):
    #     wt_w = np.dot(wt,w)
    #     wt_f = np.dot(wt,f)
    #     t = np.linalg.lstsq(wt_w, wt_f.T)
    #     return t
    # T = calcul_tension(m)
    # print('vecteur de tension = ' , T[0])



 
    #Algorithme d'optimisation
    #définition de la fonction objective qui égale à la norme du vecteur de tension
    def objective(t):
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
        t4 = t[3]
        t5 = t[4]
        t6 = t[5]
        t7 = t[6]
        t8 = t[7]
        # return np.asscalar(np.dot(wt_w[0],t.T))
        return np.sqrt(t1**2 + t2**2 + t3**2 + t4**2 + t5**2 + t6**2 + t7**2 + t8**2)
    #Valeurs initial pour cette optimisation  
    t0 =np.array([10,10,10,10,10,10,10,10])
    
    #définition des contraintes qui égale au 8 équation obtenu
    def constraint1(t):
        g = np.asscalar(np.dot(wt_w[0],t.T))
        s = np.asscalar(wt_f.T[0])
        return g-s
    def constraint2(t):
        g = np.asscalar(np.dot(wt_w[1],t.T))
        s = np.asscalar(wt_f.T[1])
        return g-s
    def constraint3(t):
        g = np.asscalar(np.dot(wt_w[2],t.T))
        s = np.asscalar(wt_f.T[2])
        return g-s
    def constraint4(t):
        g = np.asscalar(np.dot(wt_w[3],t.T))
        s = np.asscalar(wt_f.T[3])
        return g-s
    def constraint5(t):
        g = np.asscalar(np.dot(wt_w[4],t.T))
        s = np.asscalar(wt_f.T[4])
        return g-s
    def constraint6(t):
        g = np.asscalar(np.dot(wt_w[5],t.T))
        s = np.asscalar(wt_f.T[5])
        return g-s
    def constraint7(t):
        g = np.asscalar(np.dot(wt_w[6],t.T))
        s = np.asscalar(wt_f.T[6])
        return g-s
    def constraint8(t):
        g = np.asscalar(np.dot(wt_w[7],t.T))
        s = np.asscalar(wt_f.T[7])
        return g-s
    #définition des valeurs limites min et max des tensions des cables à trouver :
    b = (0.0, 200.0)
    bnds = [b,b,b,b,b,b,b,b]
    
    con1 = {'type':'eq','fun':constraint1}
    con2 = {'type':'eq','fun':constraint2}
    con3 = {'type':'eq','fun':constraint3}
    con4 = {'type':'eq','fun':constraint4}
    con5 = {'type':'eq','fun':constraint5}
    con6 = {'type':'eq','fun':constraint6}
    con7 = {'type':'eq','fun':constraint7}
    con8 = {'type':'eq','fun':constraint8}
    cons = [con1,con2,con3,con4,con5,con6,con7,con8]
    sol = minimize(objective,t0, method='SLSQP', bounds = bnds,constraints=cons)
    # on trouve la solution optimale de la distribution des tensions dans les cables selon les contraintes saisies:
    print(sol)
    
    
    #Attribution positions des sommets:   
    point1_x = s1[0]+a1[0]   
    point1_y = s1[1]+a1[1]
    point1_z = s1[2]+a1[2]
    
    point2_x = s2[0]+a2[0]
    point2_y = s2[1]+a2[1]
    point2_z = s2[2]+a2[2]
    
    point3_x = s3[0]+a3[0]
    point3_y = s3[1]+a3[1]
    point3_z = s3[2]+a3[2]    
    
    point4_x = s4[0]+a4[0]
    point4_y = s4[1]+a4[1]
    point4_z = s4[2]+a4[2] 
    
    point5_x = s5[0]+a5[0]
    point5_y = s5[1]+a5[1]
    point5_z = s5[2]+a5[2] 
    
    point6_x = s6[0]+a6[0]
    point6_y = s6[1]+a6[1]
    point6_z = s6[2]+a6[2] 
    
    point7_x = s7[0]+a7[0]
    point7_y = s7[1]+a7[1]
    point7_z = s7[2]+a7[2] 
    
    point8_x = s8[0]+a8[0]
    point8_y = s8[1]+a8[1]
    point8_z = s8[2]+a8[2] 
    
    # Simulation
    p[0] = p[0] + x_point/100
    p[1] = p[1] + y_point/100
    p[2] = p[2] + z_point/100
    
    if (phi_fin > phi or psi_fin>psi or théta_fin>théta):
        phi = phi + ((phi_point/100)*np.pi/180)
        psi =  psi + ((psi_point/100)*np.pi/180)
        théta = théta + ((théta_point/100)*np.pi/180)
    else:
        phi = phi - ((phi_point/100)*np.pi/180)
        psi =  psi - ((psi_point/100)*np.pi/180)
        théta = théta - ((théta_point/100)*np.pi/180)
    
           
    
    #Plot:
    x = [ a1[0],point1_x ,point2_x ,a2[0],point2_x ,point3_x ,a3[0],
      point3_x ,point4_x ,a4[0],point4_x , point1_x ,
      point5_x ,a5[0],point5_x ,point6_x ,a6[0],point6_x ,
      point7_x ,a7[0],point7_x ,point8_x ,a8[0],point8_x ,point4_x ,point8_x ,point5_x ,point8_x ,
      point7_x ,point3_x ,point7_x ,point6_x ,point2_x ]
    
    y =  [  a1[1],point1_y,point2_y,a2[1],point2_y,point3_y,a3[1],
          point3_y,point4_y,a4[1],point4_y, point1_y,
          point5_y,a5[1],point5_y,point6_y,a6[1],point6_y,
          point7_y,a7[1],point7_y,point8_y,a8[1],point8_y,point4_y,point8_y,point5_y,point8_y,
          point7_y,point3_y,point7_y,point6_y,point2_y ]
    
    z = [  a1[2],point1_z,point2_z,a2[2],point2_z,point3_z,a3[2],
          point3_z,point4_z,a4[2],point4_z, point1_z,
          point5_z,a5[2],point5_z,point6_z,a6[2],point6_z,
          point7_z,a7[2],point7_z,point8_z,a8[2],point8_z,point4_z,point8_z,point5_z,point8_z,
          point7_z,point3_z,point7_z,point6_z,point2_z  ]

    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label='6ddl - 8 cables')
    
    # fig = plt.figure(figsize=(8,6))
    ax1 = fig.gca(projection='3d')
    
    # plot trajectoire de la plateforme mobile
    px.append(p[0])
    py.append(p[1])
    pz.append(p[2])
    ax.scatter3D(px[0], py[0], pz[0])
    ax1.plot(px, py, pz, label='Trajectoire')
    
    # plot courbes d'évolution des longeurs des cables :
    Ls1.append(Length_s1)
    Ls2.append(Length_s2)
    Ls3.append(Length_s3)
    Ls4.append(Length_s4)
    Ls5.append(Length_s5)
    Ls6.append(Length_s6)
    Ls7.append(Length_s7)
    Ls8.append(Length_s8)    
    
    #plot des surfaces:
    x11 = np.array([[point1_x,point2_x], [point4_x,point3_x]])
    y11 = np.array([[point1_y,point2_y], [point4_y,point3_y]])
    z11 = np.array([[point1_z,point2_z],[ point4_z,point3_z]])
    
    x22 = np.array([[point1_x,point5_x],[ point2_x,point6_x]])
    y22 = np.array([[point1_y,point5_y],[ point2_y,point6_y]])
    z22 = np.array([[point1_z,point5_z],[ point2_z,point6_z]])
    
    x33 = np.array([[point2_x,point6_x],[ point3_x,point7_x]])
    y33 = np.array([[point2_y,point6_y],[ point3_y,point7_y]])
    z33 = np.array([[point2_z,point6_z],[ point3_z,point7_z]])
    
    x44 = np.array([[point1_x,point5_x],[ point4_x,point8_x]])
    y44 = np.array([[point1_y,point5_y],[ point4_y,point8_y]])
    z44 = np.array([[point1_z,point5_z],[ point4_z,point8_z]])
    
    x55 = np.array([[point3_x,point7_x],[ point4_x,point8_x]])
    y55 = np.array([[point3_y,point7_y],[ point4_y,point8_y]])
    z55 = np.array([[point3_z,point7_z],[ point4_z,point8_z]])
    
    x66 = np.array([[point5_x,point6_x], [point8_x,point7_x]])
    y66 = np.array([[point5_y,point6_y], [point8_y,point7_y]])
    z66 = np.array([[point5_z,point6_z], [point8_z,point7_z]])
    
    ax.plot_surface(x11, y11, z11,alpha=0.5)
    ax.plot_surface(x22, y22, z22,alpha=0.5)
    ax.plot_surface(x33, y33, z33,alpha=0.5)
    ax.plot_surface(x44, y44, z44,alpha=0.5)
    ax.plot_surface(x55 ,y55, z55,alpha=0.5)
    ax.plot_surface(x66, y66, z66,alpha=0.5)
    
    #plot des points sommet du cube
    ax.scatter3D(point1_x, point1_y, point1_z)
    ax.scatter3D(point2_x, point2_y, point2_z)
    ax.scatter3D(point3_x, point3_y, point3_z)
    ax.scatter3D(point4_x, point4_y, point4_z)
    ax.scatter3D(point5_x, point5_y, point5_z)
    ax.scatter3D(point6_x, point6_y, point6_z)
    ax.scatter3D(point7_x, point7_y, point7_z)
    ax.scatter3D(point8_x, point8_y, point8_z)
    ax.scatter3D(a1[0], a1[1], a1[2]), ax.scatter3D(a2[0], a2[1], a2[2]),
    ax.scatter3D(a3[0], a3[1], a3[2]), ax.scatter3D(a4[0], a4[1], a4[2]),  
    
    fig.tight_layout()
   
    #limites des axes
    ax.set_xlim3d([0.0, 20.0])
    ax.set_xlabel('x')
    
    ax.set_ylim3d([0.0, 20.0])
    ax.set_ylabel('y')
    
    ax.set_zlim3d([0.0, 20.0])
    ax.set_zlabel('z')
    ax.legend()
    
    
    #Attribution des vitesse à une liste pour le traçage des courbes de leurs évolution au cours du temps:
    s_1.append(q[0][0])    
    s_2.append(q[1][0])
    s_3.append(q[2][0])
    s_4.append(q[3][0])
    s_5.append(q[4][0])
    s_6.append(q[5][0])
    s_7.append(q[6][0])
    s_8.append(q[7][0])
    
    
    #Attribution des tensions à une liste pour le traçage des courbes de leurs évolution au cours du temps:
    n_1.append(sol.x[0])
    n_2.append(sol.x[1])
    n_3.append(sol.x[2])
    n_4.append(sol.x[3])
    n_5.append(sol.x[4])
    n_6.append(sol.x[5])
    n_7.append(sol.x[6])
    n_8.append(sol.x[7])
    
    
      
    plt.draw() #pour la mis-à-jour
    plt.pause(0.01) #temps d'affichage de nouveau plot à chaque fois



    
#Traçage de la courbe de vitesse :
k1 , k2, k3,k4,k5,k6,k7,k8 = [],[],[],[],[],[],[],[]
for i in range(len(s_1)):
     
    t1 = np.asscalar(s_1[i])
    t2 = np.asscalar(s_2[i])
    t3 = np.asscalar(s_3[i])
    t4 = np.asscalar(s_4[i])
    t5 = np.asscalar(s_5[i])
    t6 = np.asscalar(s_6[i])
    t7 = np.asscalar(s_7[i])
    t8 = np.asscalar(s_8[i])
    k1.append(t1)
    k2.append(t2)
    k3.append(t3)
    k4.append(t4)
    k5.append(t5)
    k6.append(t6)
    k7.append(t7)
    k8.append(t8)



#plot courbes des vitesses :
plt.figure(1)
t1 = np.arange(0,200,200/(len(k1)))    
ax =plt.plot(t1 , k1, label = 'vitesse linéaire du cable 1')
t2 = np.arange(0,200,200/(len(k2))) 
ax =plt.plot(t2 , k2,'o-' ,label = 'vitesse linéaire du cable 2')
t3 = np.arange(0,200,200/(len(k3))) 
ax =plt.plot(t3 , k3, label = 'vitesse linéaire du cable 3')
t4 = np.arange(0,200,200/(len(k4))) 
ax =plt.plot(t4 , k4, label = 'vitesse linéaire du cable 4')
t5 = np.arange(0,200,200/(len(k5))) 
ax =plt.plot(t5 , k5, label = 'vitesse linéaire du cable 5')
t6 = np.arange(0,200,200/(len(k6))) 
ax =plt.plot(t6 , k6, label = 'vitesse linéaire du cable 6')
t7 = np.arange(0,200,200/(len(k7))) 
ax =plt.plot(t7 , k7, label = 'vitesse linéaire du cable 7') 
t8 = np.arange(0,200,200/(len(k8)))    
ax =plt.plot(t8 , k8,'+-', label = 'vitesse linéaire du cable 8')  
 
# plt.legend(bbox_to_anchor =(0.60, 0.90), ncol = 2)
plt.legend()
plt.xlim([0,80])
plt.ylim([-80,80])
plt.xlabel('temps')
plt.ylabel('vitesse')
           
plt.title("évolution des vitesses d'enroulement/déroulement des cables")
plt.show()

#plot courbes des tensions :
plt.figure(2)
n1 = np.arange(0,400,400/len(n_1))
plt.plot(n1 , n_1 , label = 'tension du cable 1')
n2 = np.arange(0,400,400/len(n_2))
plt.plot(n2 , n_2 ,'o-' ,label = 'tension du cable 2')
n3 = np.arange(0,400,400/len(n_3))
plt.plot(n3 , n_3 , label = 'tension du cable 3')
n4 = np.arange(0,400,400/len(n_4))
plt.plot(n4 , n_4 , label = 'tension du cable 4')
n5 = np.arange(0,400,400/len(n_5))
plt.plot(n5 , n_5 , label = 'tension du cable 5')
n6 = np.arange(0,400,400/len(n_6))
plt.plot(n6 , n_6 , label = 'tension du cable 6')
n7 = np.arange(0,400,400/len(n_7))
plt.plot(n7 , n_7 , label = 'tension du cable 7')
n8 = np.arange(0,400,400/len(n_8))
plt.plot(n8 , n_8 , label = 'tension du cable 8')
plt.legend()
plt.xlim([0,80])
plt.ylim([0,80])
plt.xlabel('temps')
plt.ylabel('tension')

plt.title("évolution des tensions dans les cables")
plt.show

# plot courbes des longeurs:
plt.figure(3)
l1 = np.arange(0,100,100/len(Ls1))
plt.plot(l1 , Ls1 , label = 'Longeur du cable 1')
l2 = np.arange(0,100,100/len(Ls2))
plt.plot(l2 , Ls2 ,'o-' ,label = 'Longeur du cable 2')
l3 = np.arange(0,100,100/len(Ls3))
plt.plot(l3 , Ls3 , label = 'Longeur du cable 3')
l4 = np.arange(0,100,100/len(Ls4))
plt.plot(l4 , Ls4 , label = 'Longeur du cable 4')
l5 = np.arange(0,100,100/len(Ls5))
plt.plot(l5 , Ls5 , label = 'Longeur du cable 5')
l6 = np.arange(0,100,100/len(Ls6))
plt.plot(l6 , Ls6 , label = 'Longeur du cable 6')
l7 = np.arange(0,100,100/len(Ls7))
plt.plot(l7 , Ls7 , label = 'Longeur du cable 7')
l8 = np.arange(0,100,100/len(Ls8))
plt.plot(l8 , Ls8 , label = 'Longeur du cable 8')

plt.legend(bbox_to_anchor =(0.90, 0.90), ncol = 2)
plt.xlim([0,40])
plt.ylim([0,40])
plt.xlabel('temps')
plt.ylabel('longeur')

plt.title("évolution des longeurs des cables")
plt.show

plt.ioff()  #pour arreter la simulation
plt.show()    
