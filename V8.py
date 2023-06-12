# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:43:22 2023
@author: Arnaud Costermans
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from wavelen2rgb import wavelen2rgb #importer de http://www.johnny-lin.com/py_refs/wavelen2rgb.html

#donnée issue de https://doi.org/10.1364/AO.20.000177 (Raymond C. Smith and Karen S. Baker (1981))

Liste_longeur_onde=[200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800]
Liste_kappa=[3.07, 1.99, 1.31, 0.927, 0.72, 0.559, 0.457, 0.373, 0.288, 0.215, 0.141, 0.105, 0.0844, 0.0678, 0.0561, 0.0463, 0.0379, 0.03, 0.022, 0.0191, 0.0171, 0.0162, 0.0153, 0.0144, 0.0145, 0.0145, 0.0156, 0.0156, 0.0176, 0.0196, 0.0257, 0.0357, 0.0477, 0.0507, 0.0558, 0.0638, 0.0708, 0.0799, 0.108, 0.157, 0.244, 0.289, 0.309, 0.319, 0.329, 0.349, 0.4, 0.43, 0.45, 0.5, 0.65, 0.839, 1.169, 1.799, 2.38, 2.47, 2.55, 2.51, 2.36, 2.16, 2.07]
Liste_beta=[0.151, 0.119, 0.0995, 0.082, 0.0685, 0.0575, 0.0485, 0.0415, 0.0353, 0.0305, 0.0262, 0.0229, 0.02, 0.0175, 0.0153, 0.0134, 0.012, 0.0106, 0.0094, 0.0084, 0.0076, 0.0068, 0.0061, 0.0055, 0.0049, 0.0045, 0.0041, 0.0037, 0.0034, 0.0031, 0.0029, 0.0026, 0.0024, 0.0022, 0.0021, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0013, 0.0012, 0.0011, 0.001, 0.001, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004]

def calcul_a_b_Isotrope(w, g):
    """    
    Calcule a et b, les coeficient de la matrice dans le cas isotropique

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float
        valeur compris entre -1 et 1 qui represent le parametre d'asymetrie.

    Returns
    -------
    a : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    b : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.

    """
    
    a=(1-w*((1+g)/2))
    b=(w*((1-g)/2))
    return a, b

def calcul_a_b_c_d_non_Isotrope(w,g):
    """
    Calcule a,b,c et d, les coeficient de la matrice dans le cas non-isotropique

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : (float,float)
        Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.


    Returns
    -------
    a : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    b : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    c : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.
    d : float
        valeur compris entre 0 et 1 qui corespond a des coefficient de la matice.

    """
    e=g[0]
    f=g[1]
    a=(w*e-1)
    b=(w*(1-f))
    c=-w*(1-e)
    d=(-w*f+1)
    return a,b,c,d

def modelflux(tau, Fbas_Fhaut,w,g):
    """
    Permet de calculer Fbas' et Fhaut'. Cette fonction est uniqument appeller par solve_ivp de la librairie scipy.integrate
    
    Parameters
    ----------
    tau : float
        l'épaiseur optique.
    Fbas_Fhaut : tuple [float,float]
        Vecteur composer de Fbas et de Fhaut.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.

    Returns
    -------
    list : tuple [float,float]
        Vecteur composer de Fbas' et de Fhaut'.

    """
    
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
    else:
        a,b = calcul_a_b_Isotrope(w, g)
    Fbas, Fhaut= Fbas_Fhaut
    return [ -a*Fbas+b*Fhaut, -b*Fbas+a*Fhaut]


def graph_des_phase(w, g, color="black"):
    """
    Génère le diagrame de phase 

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    color: string or (float,float,float)
        un argument couleur de mathplotlib ou une valeur RVB. The default is "black".
    Returns
    -------
    None.

    """
    #creation des parametres de streamplot
    Y, X = np.mgrid[0:1:200j, 0:1:200j] 
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        U = a*X+b*Y 
        V = c*X+d*Y
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        U = -a*X+b*Y 
        V = -b*X+a*Y
    
    #tracage du digramme des phases
    plt.streamplot(X, Y, U, V, density = 1,arrowsize=3,color=color)

def graph_des_phase_nuage(Fbas_init, w, g, color="black"):
    """
    Génère le diagrame de phase 

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    color: string or (float,float,float)
        un argument couleur de mathplotlib ou une valeur RVB. The default is "black".
    Returns
    -------
    None.

    """
    #creation des parametres de streamplot
    Y, X = np.mgrid[0:1:200j, 0:1:200j] 
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        U = (a*X+b*Y)/Fbas_init
        V = (c*X+d*Y)
        print("here")
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        U = (-a*X+b*Y)/Fbas_init
        V = (-b*X+a*Y)/Fbas_init
    
    #tracage du digramme des phases
    plt.streamplot(X, Y, U, V, density = 1,arrowsize=3,color=color,zorder=-1)

def solution_particuliere (Fbas_init, Fhaut_init, w, g, color="black", resolution=1000, tau_min=0, tau_max=20):
    """
    Permet de tracer une solution particuliaire 

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    color: string or (float,float,float)
        un argument couleur de mathplotlib ou une valeur RVB. The default is "black".
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    None.

    """
    
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))

    #tracage de la solution particuliere
    plt.plot(solution.y[0], solution.y[1], '-',color=color, lw=3)


def point_particulier(Fbas_init, Fhaut_init, w, g, tau=0, resolution=1000, tau_min=0, tau_max=20):
    """
    
    Permet de tracer un point sur une solution particuliaire
    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    tau : int, optional
        La valeur de tau, l'epaisseur optique, a laquelle on s'arrete. The default is 0.
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    solution: OdeResult object of scipy.integrate._ivp.ivp
        Une solution particuliere continue 
        

    """
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], dense_output=True, t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))
    plt.plot(solution.sol.__call__(tau)[0], solution.sol.__call__(tau)[1], 'k.', ms=20)
    return solution
def vecteur_propre(w,g,typ='r-'):
    """
    Permet de dessiner les vecteur propre sur le graphe

    Parameters
    ----------
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    typ: string 
        Un format String conforme a la documentation Mathplotlib. The default is 'r-'.

    Returns
    -------
    None.

    """
    if type(g)==tuple:
        a,b,c,d=calcul_a_b_c_d_non_Isotrope(w, g)
        e=g[0]
        f=g[1]
    else:
        a,b = calcul_a_b_Isotrope(w, g)
        e=(g+1)/2
        f=e
    
    x_v1 = [0, -(2 - e*w - f*w + np.sqrt(4 - 4*e*w - 4*f*w - 4*(w**2) + 4*e*(w**2) + (e**2)*(w**2) + 4*f*(w**2) - 2*e*f*(w**2) + (f**2)*(w**2)))/(2*(-1 + e)*w)]
    y_v1 = [0, 1]
    x_v2 = [0,-(2 - e*w - f*w - np.sqrt(4 - 4*e*w - 4*f*w - 4*(w**2) + 4*e*(w**2) + (e**2)*(w**2) + 4*f*(w**2) - 2*e*f*(w**2) + (f**2)*(w**2)))/(2*(-1 + e)*w)]
    y_v2 = [0, 1]
    plt.plot(x_v1,y_v1,typ,lw=4,zorder=0)
    plt.plot(x_v2,y_v2,typ,lw=4,zorder=0)
    

def mise_en_forme():
    """
    Permet de mettre en forme le graphique

    Returns
    -------
    None.

    """
    
    #parametrage des axes
    label=["0","0.2","0.4","0.6","0.8","1"]
    plt.tick_params(labelsize=24)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel(r'$\frac{F_{\!\uparrow}}{F_{\!\downarrow\!0}}$', fontsize=40,  rotation='horizontal',labelpad=20.0 ,y=0.6)
    plt.xlabel(r'$\frac{F_{\!\downarrow}}{F_{\!\downarrow\!0}}$', fontsize=40,labelpad=-15.0)
    plt.xticks(np.linspace(0,1,6),label)
    plt.yticks(np.linspace(0,1,6),label)
    ax=plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

def arc_en_ceil(Fbas_init, Fhaut_init,g,longeur_onde):
    """
    Permet de tracer une solution particulier pour un w corespondant a une certain longeru d'onde.

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    longeur_onde : int
        La longeur d'onde a tracer avec son w respectif.

    Returns
    -------
    None.

    """
    
    if not(longeur_onde<380 or 780<longeur_onde):
        lamda=Liste_longeur_onde.index(longeur_onde)
        w=Liste_beta[lamda]/(Liste_beta[lamda]+Liste_kappa[lamda])
        solution_particuliere(Fbas_init, Fhaut_init, w, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[lamda])]))
         
def point_final (Fbas_init, Fhaut_init, w, g, seuil_max=1, seuil_min=0, resolution=1000, tau_min=0, tau_max=20):
    """
    Cette fonction permet de trouver le point sur une solution particulier qui dépasse un certain seuil de Fhaut

    Parameters
    ----------
    Fbas_init : float
        La valeur initial de Fbas. Compris entre 0 et 1.
    Fhaut_init : float
        La valeur initial de Fhaut. Compris entre 0 et 1.
    w : float
        valeur compris entre 0 et 1 qui represent l'albedo simple.
    g : float or (float,float)
        Soit valeur compris entre -1 et 1 qui represent le parametre d'asymetrie 
        ou un Tuple contenant p_bas,bas et p_haut,haut compris entre 0 et 1.
    seuil_max : float, optional
        Le seuil supérieur au-delà duquel on arrêt la solution. The default is 1.
    seuil_min : float, optional
        Le seuil inférieur en dessous duquel on arrêt la solution. The default is 0.
    resolution : int, optional
        Le nombre de point qui seront tracer pour la solution particuliaire. The default is 1000.
    tau_min : int, optional
        La valuer initial pour la derivation par tau, l'epaiseur optique. The default is 0.
    tau_max : int, optional
        La valuer final pour la derivation par tau, l'epaiseur optique. The default is 20.

    Returns
    -------
    tau : float
        La valeur de tau au seuil où on arrêt la solution.
    Fbas : float
        La valeur de Fbas au seuil où on arrêt la solution.
    Fhaut : float
        La valeur de Fhaut au seuil où on arrêt la solution.

    """
    
    
    #calcule de la solution particuliaire
    solution=solve_ivp(modelflux, [tau_min,tau_max], [Fbas_init,Fhaut_init], dense_output=True, t_eval=np.linspace(tau_min,tau_max,resolution), args=(w,g))
    for loop in range (len(solution.t)):
        if solution.y[1][loop]>seuil_max:
            if abs(seuil_max-solution.y[1][loop])<abs(seuil_max-solution.y[1][loop-1]):
                endpoint=loop
                break
            else:
                endpoint=loop-1
                break
        elif solution.y[1][loop]<seuil_min:
            if abs(seuil_min-solution.y[1][loop])<abs(seuil_min-solution.y[1][loop-1]):
                endpoint=loop
                break
            else:
                endpoint=loop-1
                break
        endpoint=loop
    Fbas=solution.y[0][endpoint]
    Fhaut=solution.y[1][endpoint]
    tau=solution.t[endpoint]
    return tau, Fbas , Fhaut


#definiton des variable global

w=0.9 #definition de l'albedo simple
g=-1 #definition du parametre d'asymetrie

# creation de la figure
fig = plt.figure(figsize = (12, 7))
mise_en_forme()

"""arc en ceil"""
tau_final=[]
w_final=[]
z_final=[]

for loop in range (len(Liste_longeur_onde)):
    if not(Liste_longeur_onde[loop]<400 or 780<Liste_longeur_onde[loop]):
        arc_en_ceil(1, 0.00008, g, Liste_longeur_onde[loop])
        lamda=Liste_longeur_onde.index(Liste_longeur_onde[loop])
        w=Liste_beta[lamda]/(Liste_beta[lamda]+Liste_kappa[lamda])
        tau_final.append(round(point_final(1, 0.2, w, g)[0],2))
        w_final.append(w)
        z_final.append((point_final(1, 0.2, w, g)[0])/(Liste_beta[lamda]+Liste_kappa[lamda]))
lamda=Liste_longeur_onde.index(400)
w=Liste_beta[lamda]/(Liste_beta[lamda]+Liste_kappa[lamda])
vecteur_propre(w, g,'k-')
lamda=Liste_longeur_onde.index(780)
w=Liste_beta[lamda]/(Liste_beta[lamda]+Liste_kappa[lamda])
vecteur_propre(w, g,'k-')
print(tau_final)
plt.ylim(0,0.0001)

"""Longeur d'onde aire"""
# kappa_450=0.02
# kappa_675=kappa_450/5
# beta_450=0.1
# beta_675=beta_450/5
# w_450=beta_450/(beta_450+kappa_450)
# w_675=beta_675/(beta_675+kappa_675)
# vecteur_propre(w_450, g,typ="black")
# vecteur_propre(w_675, g,typ="black")
# graph_des_phase(w_450, g, "#1f77b4")
# tau=0.25
# solution_particuliere(1, 0.30, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
# solution_particuliere(1, 0.30, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
# solution_particuliere(1, 0.57, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
# solution_particuliere(1, 0.57, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
# solution_particuliere(0, 0.50, w_450, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(450)])]),tau_max=5*tau)
# solution_particuliere(0, 0.50, w_675, g, color=tuple([float(x/100) for x in wavelen2rgb(Liste_longeur_onde[Liste_longeur_onde.index(670)])]),tau_max=tau)
# point_particulier(1, 0.15, w_450, g, 0.40040040040040037)
# print (point_final(1, 0.15, w_450, g))
# print (point_final(1, 0.15, w_675, g))
# print(w_450,w_675)
 
"""graphe de base"""
# vecteur_propre(w, g)
# graph_des_phase(w, g, "#1f77b4")
# solution_particuliere(1, 0.5, w, g)
# point_particulier(1, 0.5, w, g, 1.25)

"""graphe nuage"""
#graph_des_phase(w, g, "#1f77b4")  
# graph_des_phase_nuage(0.4, w, g, "#1f77b4")
# plt.fill_between([0,1], [0,1],[1,1],facecolor="w", hatch="/", edgecolor="k", linewidth=0.0)
# plt.plot([0,1],[0,1],'k--')
# vecteur_propre(w, g,'b')
# solution_particuliere(1, 0.6, w, g,tau_max=1.5)
# tau,albedoX,albedoY=point_final(1, 0.6, w, g,tau_max=1.5)
#plt.plot([0,albedoX*3],[0,albedoY*3],color=(0,1,0),lw=4)

# montrer la figure
plt.show()

fig = plt.figure(figsize = (12, 7))
ax1=plt.gca()
ax2 = ax1.twinx()
ax1.plot(Liste_longeur_onde,Liste_kappa)
ax2.plot(Liste_longeur_onde,Liste_beta)


